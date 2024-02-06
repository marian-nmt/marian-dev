#include "common/io.h"

#include "3rd_party/cnpy/cnpy.h"
#include "common/definitions.h"
#include "common/shape.h"
#include "common/types.h"

#include "common/binary.h"
#include "common/io_item.h"

#include "training/communicator.h"

namespace marian {
namespace io {

bool isNpz(const std::string& fileName) {
  return fileName.size() >= 4
         && fileName.substr(fileName.length() - 4) == ".npz";
}

bool isBin(const std::string& fileName) {
  return fileName.size() >= 4
         && fileName.substr(fileName.length() - 4) == ".bin";
}

ModelWeights::FileType ModelWeights::getFileType(const std::string& fileName) {
  if(isNpz(fileName))
    return FileType::isNpz;
  else if(isBin(fileName))
    return FileType::isBin;
  else
    ABORT("Unknown file format for file {}", fileName);
}

std::vector<Item>& ModelWeights::items() {
  load();
  return items_;
}

const std::vector<Item>& ModelWeights::items() const {
  const_cast<ModelWeights&>(*this).load();
  return items_;
}

const void* ModelWeights::data() const {
  const_cast<ModelWeights&>(*this).load();
  switch (fileType_) {
    case FileType::isNpz:
      return nullptr;
    case FileType::isBin:
      return mmap_->data();
    case FileType::isBuf:
      return ptr_;
    case FileType::isDummy:
      ABORT("Cannot get data from dummy model");
    default:
      ABORT("Unknown file type");
  }
}

size_t ModelWeights::size() const {
  const_cast<ModelWeights&>(*this).load();
  switch (fileType_) {
    case FileType::isNpz:
      return 0;
    case FileType::isBin:
      return mmap_->size();
    case FileType::isBuf:
      ABORT("Cannot get size of buffer");
    case FileType::isDummy:
      ABORT("Cannot get size from dummy model");
    default:
      ABORT("Unknown file type");
  }
}

// @TODO: bring back fast peeking into the file to get config
// Load YAML from item
YAML::Node ModelWeights::getYamlFromModel(const std::string& varName) const {
  const_cast<ModelWeights&>(*this).load();
  for(auto& item : items_) {
    if(item.name == varName) {
      return YAML::Load(item.data());
    }
  }
  return YAML::Node();
}

void loadItemsFromNpz(const std::string& fileName, std::vector<Item>& items) {
  auto numpy = cnpy::npz_load(fileName);
  for(auto it : numpy) {
    ABORT_IF(it.second->fortran_order, "Numpy item '{}' is not stored in row-major order", it.first);

    Shape shape;
    shape.resize(it.second->shape.size());
    for(size_t i = 0; i < it.second->shape.size(); ++i)
      shape.set(i, (size_t)it.second->shape[i]);

    Item item;
    item.name = it.first;
    item.shape = shape;

    char npzType = it.second->type;
    int wordSize = it.second->word_size;
    if     (npzType == 'f' && wordSize == 2) item.type = Type::float16;
    else if(npzType == 'f' && wordSize == 4) item.type = Type::float32;
    else if(npzType == 'f' && wordSize == 8) item.type = Type::float64;
    else if(npzType == 'i' && wordSize == 1) item.type = Type::int8;
    else if(npzType == 'i' && wordSize == 2) item.type = Type::int16;
    else if(npzType == 'i' && wordSize == 4) item.type = Type::int32;
    else if(npzType == 'i' && wordSize == 8) item.type = Type::uint64;
    else if(npzType == 'u' && wordSize == 1) item.type = Type::uint8;
    else if(npzType == 'u' && wordSize == 2) item.type = Type::uint16;
    else if(npzType == 'u' && wordSize == 4) item.type = Type::uint32;
    else if(npzType == 'u' && wordSize == 8) item.type = Type::uint64;
    else ABORT("Numpy item '{}' type '{}' with size {} not supported", it.first, npzType, wordSize);

    item.bytes.swap(it.second->bytes);
    items.emplace_back(std::move(item));
  }
}

std::vector<Item> ModelWeights::loadItems(const std::string& fileName) {
  std::vector<Item> items;
  if(isNpz(fileName)) {
    loadItemsFromNpz(fileName, items);
  } else if(isBin(fileName)) {
    binary::loadItems(fileName, items);
  } else {
    ABORT("Unknown model file format for file {}", fileName);
  }

  return items;
}

std::vector<Item> ModelWeights::mmapItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, true);
  return items;
}

std::unique_ptr<std::lock_guard<std::mutex>> ModelWeights::scopedLockGuard() const {
  // @TODO: this should use std::optional, but as long as we use CUDA 10.x there may be
  // random problems with std::optional and nvcc compilation
  if(locking_)
    return std::unique_ptr<std::lock_guard<std::mutex>>(new std::lock_guard<std::mutex>(mutex_));
  else
    return nullptr;
}

void ModelWeights::load() {
  auto optionalLock = scopedLockGuard();

  if(loaded_)
    return;

  switch (fileType_) {
    case FileType::isNpz:
      loadItemsFromNpz(fileName_, items_);
      break;
    case FileType::isBin:
      if(mmapMode_ == MmapMode::DontMmap) {
        binary::loadItems(fileName_, items_);
      } else {
        try {
          mmap_.reset(new mio::mmap_source(fileName_));
          binary::loadItems(mmap_->data(), items_, /*mapped=*/true);
        } catch(const MarianRuntimeException& e) {
          if(mmapMode_ == MmapMode::RequiredMmap)
            ABORT("Could not memory-map file '{}': {}", fileName_, e.what());
          else
            LOG(warn, "[warning] Could not memory-map file '{}' ({}), falling back to reading from disk", fileName_, e.what());
          mmapMode_ = MmapMode::DontMmap;
          binary::loadItems(fileName_, items_);
        }
      }
      break;
    case FileType::isBuf:
      binary::loadItems(ptr_, items_, /*mapped=*/mmapMode_ != MmapMode::DontMmap);
      break;
    case FileType::isDummy:
      ABORT("Cannot load from dummy model");
    default:
      ABORT("Unknown file type");
  }

  loaded_ = true;
}

void ModelWeights::loadAndSync(Ptr<IMPIWrapper> mpi) {
  ABORT_IF(!mpi, "MPI wrapper is null");
  ABORT_IF(mmapMode_ != MmapMode::DontMmap, "Mmapping not allowed");

  if(mpi->isMainProcess())
    load();

  mpi->bCast(fileName_);
  mpi->bCast(&fileType_, 1, mpi->getDataType((size_t*)&fileType_));
  mpi->bCast(&loaded_,   1, mpi->getDataType(&loaded_));
  mpi->bCast(items_);
}

// @TODO: make cnpy and our wrapper talk to each other in terms of types
// or implement our own saving routines for npz based on npy, probably better.
void saveItemsNpz(const std::string& fileName, const std::vector<Item>& items) {
  std::vector<cnpy::NpzItem> npzItems;
  for(auto& item : items) {
    std::vector<unsigned int> shape(item.shape.begin(), item.shape.end());
    char type;

    if     (item.type == Type::float16) type = cnpy::map_type(typeid(float)); // becomes 'f', correct size is given below
    else if(item.type == Type::float32) type = cnpy::map_type(typeid(float));
    else if(item.type == Type::float64) type = cnpy::map_type(typeid(double));
    else if(item.type == Type::int8)    type = cnpy::map_type(typeid(int8_t));
    else if(item.type == Type::int16)   type = cnpy::map_type(typeid(int16_t));
    else if(item.type == Type::int32)   type = cnpy::map_type(typeid(int32_t));
    else if(item.type == Type::int64)   type = cnpy::map_type(typeid(int64_t));
    else if(item.type == Type::uint8)   type = cnpy::map_type(typeid(uint8_t));
    else if(item.type == Type::uint16)  type = cnpy::map_type(typeid(uint16_t));
    else if(item.type == Type::uint32)  type = cnpy::map_type(typeid(uint32_t));
    else if(item.type == Type::uint64)  type = cnpy::map_type(typeid(uint64_t));
    else ABORT("Other types ({}) not supported", item.type);

    npzItems.emplace_back(item.name, item.bytes, shape, type, sizeOf(item.type));
  }
  cnpy::npz_save(fileName, npzItems);
}

void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items) {
  Item item;
  item.name = varName;

  // increase size by 1 to add \0
  item.shape = Shape({(int)meta.size() + 1});

  item.bytes.resize(item.shape.elements());
  std::copy(meta.begin(), meta.end(), item.bytes.begin());
  // set string terminator
  item.bytes.back() = '\0';

  item.type = Type::int8;

  items.push_back(item);
}

void saveItems(const std::string& fileName, const std::vector<Item>& items) {
  if(isNpz(fileName)) {
    saveItemsNpz(fileName, items);
  } else if(isBin(fileName)) {
    binary::saveItems(fileName, items);
  } else {
    ABORT("Unknown file format for file {}", fileName);
  }
}

}  // namespace io
}  // namespace marian
