#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "3rd_party/mio/mio.hpp"
#include "3rd_party/yaml-cpp/yaml.h"
#include "common/definitions.h"
#include "common/io_item.h"

#include <string>
#include <vector>

// interface for handling model files in marian, both *.npz files and
// *.bin files have the same way of accessing them and are identified
// by suffixes (*.npz or *.bin).

// Files with the *.bin suffix are supposed to be memory-mappable for
// CPU decoding.

namespace marian {

struct IMPIWrapper;

namespace io {

enum struct MmapMode { OpportunisticMmap, DontMmap, RequiredMmap };

bool isNpz(const std::string& fileName);
bool isBin(const std::string& fileName);

class ModelWeights {
private:
  std::mutex mutex_;

  std::string fileName_;
  const void* ptr_{nullptr};

  enum struct FileType : size_t { isNpz, isBin, isBuf, isDummy };
  FileType fileType_{FileType::isNpz};
  FileType getFileType(const std::string& fileName);

  MmapMode mmapMode_{MmapMode::OpportunisticMmap};

  bool loaded_{false};

  std::vector<Item> items_;
  std::unique_ptr<mio::mmap_source> mmap_;

  std::vector<Item> loadItems(const std::string& fileName);
  std::vector<Item> mmapItems(const void* ptr);

  void load();

public:
  ModelWeights(const std::string& fileName, MmapMode mmapMode = MmapMode::OpportunisticMmap)
  : fileName_(fileName), fileType_(getFileType(fileName)), mmapMode_(mmapMode) {

    // NPZ files cannot be memory-mapped, so we switch opportunistic mmap off, but keep any other mmap mode
    if(fileType_ == FileType::isNpz && mmapMode_ == MmapMode::OpportunisticMmap)
      mmapMode_ = MmapMode::DontMmap;

    // so we can croak here for NPZ files if the user sets mmap to required
    ABORT_IF(fileType_ == FileType::isNpz && mmapMode_ != MmapMode::DontMmap, "NPZ files cannot be memory-mapped");
  }

  ModelWeights(const void* ptr, MmapMode mmapMode = MmapMode::RequiredMmap)
  : ptr_(ptr), fileType_(FileType::isBuf), mmapMode_(mmapMode) {}

  ModelWeights()
  : fileType_(FileType::isDummy), mmapMode_{MmapMode::DontMmap} {}

  ModelWeights(const ModelWeights&&) = delete;
  ModelWeights(const ModelWeights&) = delete;

  std::vector<Item>& items();
  const std::vector<Item>& items() const;

  MmapMode mmapMode() const {
    return mmapMode_;
  }
  const void* data() const;
  size_t size() const;

  YAML::Node getYamlFromModel(const std::string& varName = "special:model.yml") const;

  void loadAndSync(Ptr<IMPIWrapper> mpi);
};

// for saving we keep the old interface since there is no intelligence going on here and it is useful
// to be able to assemble a set of items in different places.
void addMetaToItems(const std::string& meta,
                    const std::string& varName,
                    std::vector<io::Item>& items);

void saveItems(const std::string& fileName, const std::vector<Item>& items);

/**
 * Creates a flat io::Item from a given std::vector so that it can be saved in a npz file
 * or Marian's native binary format with the given name.
 */
template <typename T>
Item fromVector(const std::vector<T>& vec, const std::string& name) {
  Item item;
  item.name = std::move(name);
  item.shape = Shape({1, (int)vec.size()}); // @TODO: review if this should be {1, size} or rather just {size}
  item.type = typeId<T>();
  item.bytes.resize(vec.size() * sizeOf(item.type));
  std::copy((char*)vec.data(), (char*)(vec.data() + vec.size()), item.bytes.begin());
  return item;
}

}  // namespace io
}  // namespace marian
