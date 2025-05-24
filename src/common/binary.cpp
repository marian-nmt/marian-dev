#include "common/binary.h"
#include "common/definitions.h"
#include "common/file_stream.h"
#include "common/io_item.h"
#include "common/types.h"
#include "tensors/cpu/integer_common.h"

#if USE_SSL
#include "common/config.h"
#include "common/crypt.h"
#endif

#include <string>

namespace marian {
namespace io {

namespace binary {

struct Header {
  uint64_t nameLength;
  uint64_t type;
  uint64_t shapeLength;
  uint64_t dataLength;
};

// cast current void pointer to T pointer and move forward by num elements
template <typename T>
const T* get(const void*& current, uint64_t num = 1) {
  const T* ptr = (const T*)current;
  current = (const T*)current + num;
  return ptr;
}

void loadItems(const void* current, std::vector<io::Item>& items, bool mapped) {
  uint64_t binaryFileVersion = *get<uint64_t>(current);

  const void *headerCurrent = nullptr;

  bool encrypted = false;
  std::string decryptedHeader;
  if(binaryFileVersion == BINARY_FILE_VERSION) {
    headerCurrent = current;
#if USE_SSL
  } else if(binaryFileVersion == BINARY_FILE_VERSION_WITH_ENCRYPTION) {
    LOG(info, "Loading encrypted model");
    encrypted = true;
    uint64_t headerLen = *get<uint64_t>(current);
    std::string encryptedHeader(get<char>(current, headerLen), headerLen);
    std::string encryptionKey = Config::encryptionKey;
    ABORT_IF(encryptionKey.empty(), "No encryption key provided or set");

    decryptedHeader = marian::crypt::decrypt_aes_256_gcm(encryptedHeader, encryptionKey);
    headerCurrent = decryptedHeader.data();
#endif
  } else {
    ABORT("Unknown binary file version {}", binaryFileVersion);
  }

  ABORT_IF(headerCurrent == nullptr, "Header not loaded");

  uint64_t numHeaders = *get<uint64_t>(headerCurrent); // number of item headers that follow
  const Header* headers = get<Header>(headerCurrent, numHeaders); // read that many headers

  // prepopulate items with meta data from headers
  items.resize(numHeaders);
  for(int i = 0; i < numHeaders; ++i) {
    items[i].type = (Type)headers[i].type;
    items[i].name = get<char>(headerCurrent, headers[i].nameLength);
    items[i].mapped = mapped;
  }

  // read in actual shape and data
  for(int i = 0; i < numHeaders; ++i) {
    uint64_t len = headers[i].shapeLength;
    items[i].shape.resize(len);
    const int* arr = get<int>(headerCurrent, len); // read shape
    std::copy(arr, arr + len, items[i].shape.begin()); // copy to Item::shape
  }

  if(!encrypted)
    current = headerCurrent;

  // move by offset bytes, aligned to 256-bytes boundary
  uint64_t offset = *get<uint64_t>(current);
  get<char>(current, offset);

  for(int i = 0; i < numHeaders; ++i) {
    // For intgemm AVX512 and AVX512VNNI have the same arangement, but the VNNI algorithm is faster.
    // Change the type to the fastest one supported.
    if (items[i].type == Type::intgemm8avx512) {
      items[i].type = cpu::integer::getIntgemmType(Type::intgemm8);
    }
    if(items[i].mapped) { // memory-mapped, hence only set pointer
      if(items[i].type == Type::intgemm8 || items[i].type == Type::intgemm16)
        throw MarianRuntimeException("mmap format not supported for hardware non-specific intgemm matrices", getCallStack(/*skipLevels=*/0));
      items[i].ptr = get<char>(current, headers[i].dataLength);
    } else { // reading into item data
      uint64_t len = headers[i].dataLength;
      items[i].bytes.resize(len);
      const char* ptr = get<char>(current, len);
      // Intgemm8/16 matrices in binary model are just quantized, however they also need to be reordered
      // Reordering depends on the architecture (SSE/AVX2/AVX512) so we read in the quantized matrices and
      // then reorder them before adding them as a parameter in the graph.
      if (matchType<intgemm8>(items[i].type)) {
        items[i].type = cpu::integer::getIntgemmType(Type::intgemm8);
        cpu::integer::prepareAndTransposeB<Type::intgemm8>(items[i], ptr);
      } else if (matchType<intgemm16>(items[i].type)) {
        items[i].type = cpu::integer::getIntgemmType(Type::intgemm16);
        cpu::integer::prepareAndTransposeB<Type::intgemm16>(items[i], ptr);
      } else {
        std::copy(ptr, ptr + len, items[i].bytes.begin());
      }
    }
  }
}

void loadItems(const std::string& fileName, std::vector<io::Item>& items) {
  // Read file into buffer
  uint64_t fileSize = filesystem::fileSize(fileName);
  std::vector<char> buf(fileSize);
  io::InputFileStream in(fileName);
  in.read(buf.data(), buf.size());

  // Load items from buffer without mapping
  loadItems(buf.data(), items, false);
}

io::Item getItem(const void* current, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(current, items, /*mapped=*/true);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}

io::Item getItem(const std::string& fileName, const std::string& varName) {
  std::vector<io::Item> items;
  loadItems(fileName, items);

  for(auto& item : items)
    if(item.name == varName)
      return item;

  return io::Item();
}


// append binary data to a vector of chars based on the type
template <typename T>
void append(std::string& buf, const T* ptr, size_t num = 1) {
  buf.append(reinterpret_cast<const char*>(ptr), num * sizeof(T));
}

std::string createBinaryHeader(std::vector<const io::Item*>& items) {
  std::string buffer;

#if USE_SSL
  std::string encryptionKey = Config::encryptionKey;
  bool encrypted = !encryptionKey.empty();

  if(encrypted) {
    // if encrypted shuffle items to avoid leaking order of parameters
    // use cryptographically secure random number generator
    marian::crypt::OpenSSLRNG rng;
    // this will change the order of items outside of this function, too, as intended.
    std::shuffle(items.begin(), items.end(), rng);
  }
#else
  bool encrypted = false;
#endif

  uint64_t binaryFileVersion = encrypted ? BINARY_FILE_VERSION_WITH_ENCRYPTION : BINARY_FILE_VERSION;
  append(buffer, &binaryFileVersion);

  std::string headerBuffer;
  std::vector<Header> headers;
  for(const auto& item : items) {
    headers.push_back(Header{item->name.size() + 1,
                             (uint64_t)item->type,
                             item->shape.size(),
                             item->bytes.size()}); // binary item size with padding, will be 256-byte-aligned
  }

  uint64_t headerSize = headers.size();
  append(headerBuffer, &headerSize);
  append(headerBuffer, headers.data(), headers.size());

  // Write out all names
  for(const auto& item : items)
    append(headerBuffer, item->name.data(), item->name.size() + 1);

  // Write out all shapes
  for(const auto& item : items)
    append(headerBuffer, item->shape.data(), item->shape.size());

  if(encrypted) {
#if USE_SSL
    LOG(info, "Saving encrypted model");
    std::string encryptedHeaderBuffer = marian::crypt::encrypt_aes_256_gcm(headerBuffer, encryptionKey);
    uint64_t encryptedHeaderSize = encryptedHeaderBuffer.size();
    append(buffer, &encryptedHeaderSize);
    append(buffer, encryptedHeaderBuffer.data(), encryptedHeaderBuffer.size());
#else
    ABORT("Encryption requested but not supported");
#endif
  } else {
    append(buffer, headerBuffer.data(), headerBuffer.size());
  }

  return buffer;
}

void saveItems(const std::string& fileName,
               const std::vector<io::Item>& items) {
  io::OutputFileStream out(fileName);

  // create a vector of pointers to items and shuffle them (pointers, not items, to avoid copying the data)
  std::vector<const io::Item*> weakItems;
  for(auto& item : items)
    weakItems.push_back(&item);

  std::string headerBuffer = createBinaryHeader(weakItems);

  uint64_t pos = 0;
  pos += out.write(headerBuffer.data(), headerBuffer.size());

  // align to next 256-byte boundary
  uint64_t nextpos = ((pos + sizeof(uint64_t)) / 256 + 1) * 256;
  uint64_t offset = nextpos - pos - sizeof(uint64_t);

  // Write offset
  pos += out.write(&offset);
  // Write padding
  for(uint64_t i = 0; i < offset; i++) {
    char padding = 0;
    pos += out.write(&padding);
  }

  // Write out all values
  for(const auto& item : weakItems) {
    ABORT_IF(item->bytes.size() % 256 != 0,
          "Binary item size ({}) is not 256-byte aligned for tensor {}",
          item->bytes.size(),
          item->name);
    pos += out.write(item->data(), item->bytes.size()); // writes out data with padding, keeps 256-byte boundary.
                                                        // Amazingly this is binary-compatible with V1 and aligned and
                                                        // non-aligned models can be read with the same procedure.
                                                        // No version-bump required. Gets 5-8% of speed back when mmapped.
  }
}

}  // namespace binary
}  // namespace io
}  // namespace marian
