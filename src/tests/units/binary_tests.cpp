#include "catch.hpp"
#include "common/binary.h"
#include "common/config.h"
#include "common/file_stream.h"

#include "3rd_party/mio/mio.hpp"

#if USE_SSL
#include "common/crypt.h"
#endif

using namespace marian;

TEST_CASE("a few operations on binary files", "[binary]") {

  SECTION("Save two items to temporary binary file and then load and map") {

    // Create a temporary file that we will only use for the file name
    io::TemporaryFile temp("/tmp/", /*earlyUnlink=*/false);
    io::Item item1, item2;

    {
      std::vector<float>    v1 = { 3.14, 2.71, 1.0, 0.0, 1.41 };
      std::vector<uint16_t> v2 = { 5, 4, 3, 2, 1, 0 };

      item1.name  = "item1";
      item1.shape = { 5, 1 };
      item1.type  = Type::float32;
      item1.bytes.resize(v1.size() * sizeof(float));
      // resize to multiple of 256 bytes
      size_t multiplier = (size_t)ceil((float)item1.bytes.size() / (float)256);
      item1.bytes.resize(multiplier * 256);

      std::copy((char*)v1.data(), (char*)v1.data() + v1.size() * sizeof(float), item1.bytes.data());

      item2.name  = "item2";
      item2.shape = { 2, 3 };
      item2.type = Type::uint16;
      item2.bytes.resize(v2.size() * sizeof(uint32_t));
      // resize to multiple of 256 bytes
      multiplier = (size_t)ceil((float)item2.bytes.size() / (float)256);
      item2.bytes.resize(multiplier * 256);

      std::copy((char*)v2.data(), (char*)v2.data() + v2.size() * sizeof(uint16_t), item2.bytes.data());

      std::vector<io::Item> items = {item1, item2};
      io::binary::saveItems(temp.getFileName(), items);
    }

    { // test loading
      std::vector<io::Item> items;
      io::binary::loadItems(temp.getFileName(), items);

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );

      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }

    { // test mmapping
      mio::mmap_source mmap(temp.getFileName());

      // magic number at the beginning of the file
      uint64_t binaryFileVersion = *reinterpret_cast<const uint64_t*>(mmap.data());
      CHECK( binaryFileVersion == BINARY_FILE_VERSION );

      std::vector<io::Item> items;
      io::binary::loadItems(mmap.data(), items, /*mapped=*/true);

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );

      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }
  }

#if USE_SSL
  SECTION("Save two items to temporary binary file and then load and map but with encryption") {

    // Create a temporary file that we will only use for the file name
    io::TemporaryFile temp("/tmp/", /*earlyUnlink=*/false);
    io::Item item1, item2;

    // 32-byte long key for AES encryption
    Config::encryptionKey = marian::crypt::sha256("This is a test key for encryption");

    {
      std::vector<float>    v1 = { 3.14, 2.71, 1.0, 0.0, 1.41 };
      std::vector<uint16_t> v2 = { 5, 4, 3, 2, 1, 0 };

      item1.name  = "item1";
      item1.shape = { 5, 1 };
      item1.type  = Type::float32;
      item1.bytes.resize(v1.size() * sizeof(float));
      // resize to multiple of 256 bytes
      size_t multiplier = (size_t)ceil((float)item1.bytes.size() / (float)256);
      item1.bytes.resize(multiplier * 256);
      std::copy((char*)v1.data(), (char*)v1.data() + v1.size() * sizeof(float), item1.bytes.data());

      item2.name  = "item2";
      item2.shape = { 2, 3 };
      item2.type = Type::uint16;
      item2.bytes.resize(v2.size() * sizeof(uint32_t));
      // resize to multiple of 256 bytes
      multiplier = (size_t)ceil((float)item2.bytes.size() / (float)256);
      item2.bytes.resize(multiplier * 256);
      std::copy((char*)v2.data(), (char*)v2.data() + v2.size() * sizeof(uint16_t), item2.bytes.data());

      std::vector<io::Item> items = {item1, item2};
      io::binary::saveItems(temp.getFileName(), items);
    }

    { // test loading
      std::vector<io::Item> items;
      io::binary::loadItems(temp.getFileName(), items);

      // sort items by name to make sure the order is the same as in the original items
      std::sort(items.begin(), items.end(), [](const io::Item& a, const io::Item& b) {
        return a.name < b.name;
      });

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );

      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }

    { // test mmapping
      mio::mmap_source mmap(temp.getFileName());

      // magic number at the beginning of the file
      uint64_t binaryFileVersion = *reinterpret_cast<const uint64_t*>(mmap.data());
      CHECK( binaryFileVersion == BINARY_FILE_VERSION_WITH_ENCRYPTION );

      std::vector<io::Item> items;
      io::binary::loadItems(mmap.data(), items, /*mapped=*/true);

      // sort items by name to make sure the order is the same as in the original items
      std::sort(items.begin(), items.end(), [](const io::Item& a, const io::Item& b) {
        return a.name < b.name;
      });

      CHECK( item1.name == items[0].name );
      CHECK( item2.name == items[1].name );

      CHECK( std::equal(item1.data(), item1.data() + item1.size(), items[0].data()) );
      CHECK( std::equal(item2.data(), item2.data() + item2.size(), items[1].data()) );
    }
  }
#endif // USE_SSL
}
