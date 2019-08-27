#include "common/io.h"

#include "3rd_party/cnpy/cnpy.h"
#include "common/shape.h"
#include "common/types.h"

#include "common/binary.h"
#include "common/io_item.h"

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

void getYamlFromNpz(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = cnpy::npz_load(fileName, varName);
  if(item->size() > 0)
    yaml = YAML::Load(item->data());
}

void getYamlFromBin(YAML::Node& yaml,
                    const std::string& varName,
                    const std::string& fileName) {
  auto item = binary::getItem(fileName, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const std::string& fileName) {
  if(io::isNpz(fileName)) {
    io::getYamlFromNpz(yaml, varName, fileName);
  } else if(io::isBin(fileName)) {
    io::getYamlFromBin(yaml, varName, fileName);
  } else {
    ABORT("Unknown model file format for file {}", fileName);
  }
}

void getYamlFromModel(YAML::Node& yaml,
                      const std::string& varName,
                      const void* ptr) {
  auto item = binary::getItem(ptr, varName);
  if(item.size() > 0)
    yaml = YAML::Load(item.data());
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


static std::unordered_map<std::string, float> alham_map({
{"Wemb", 0.9296592},
{"decoder_l1_context_Wk", 0.99786574},
{"decoder_l1_context_Wo", 0.53925306},
{"decoder_l1_context_Wq", 0.70678043},
{"decoder_l1_context_Wv", 0.4280936},
{"decoder_l1_ffn_W1", 1.1141714},
{"decoder_l1_ffn_W2", 1.229842},
{"decoder_l1_rnn_W", 0.70819724},
{"decoder_l1_rnn_Wf", 0.78009325},
{"encoder_l1_ffn_W1", 0.45678806},
{"encoder_l1_ffn_W2", 0.8775526},
{"encoder_l1_self_Wk", 1.1568298},
{"encoder_l1_self_Wo", 0.17897393},
{"encoder_l1_self_Wq", 1.2465283},
{"encoder_l1_self_Wv", 0.20422521},
{"encoder_l2_ffn_W1", 0.5061755},
{"encoder_l2_ffn_W2", 1.2318062},
{"encoder_l2_self_Wk", 0.6026684},
{"encoder_l2_self_Wo", 0.46921992},
{"encoder_l2_self_Wq", 0.7555442},
{"encoder_l2_self_Wv", 0.23997004},
{"encoder_l3_ffn_W1", 0.44958767},
{"encoder_l3_ffn_W2", 0.7768512},
{"encoder_l3_self_Wk", 0.5133644},
{"encoder_l3_self_Wo", 0.29558635},
{"encoder_l3_self_Wq", 0.47287956},
{"encoder_l3_self_Wv", 0.23979305},
{"encoder_l4_ffn_W1", 0.4244445},
{"encoder_l4_ffn_W2", 0.71431214},
{"encoder_l4_self_Wk", 0.47557107},
{"encoder_l4_self_Wo", 0.408042},
{"encoder_l4_self_Wq", 0.41991013},
{"encoder_l4_self_Wv", 0.2435633},
{"encoder_l5_ffn_W1", 0.7091122},
{"encoder_l5_ffn_W2", 1.621571},
{"encoder_l5_self_Wk", 0.59227574},
{"encoder_l5_self_Wo", 0.7173325},
{"encoder_l5_self_Wq", 0.5866978},
{"encoder_l5_self_Wv", 0.27250895},
{"encoder_l6_ffn_W1", 0.40217918},
{"encoder_l6_ffn_W2", 2.3296742},
{"encoder_l6_self_Wk", 0.46993786},
{"encoder_l6_self_Wo", 0.7933633},
{"encoder_l6_self_Wq", 0.48293704},
{"encoder_l6_self_Wv", 0.2542558},
});

static float toFloat(uint8_t x, float scale) {
  if (x >= 8)
    return scale * std::pow(2.0, (int) x - 15);
  return -scale * std::pow(2.0, (int) x - 7);
}


void loadItemsFromNpz(const std::string& fileName, std::vector<Item>& items) {
  auto numpy = cnpy::npz_load(fileName);
  for(auto it : numpy) {
    Shape shape;
    if(it.second->shape.size() == 1) {
      shape.resize(2);
      shape.set(0, 1);
      shape.set(1, (size_t)it.second->shape[0]);
    } else {
      shape.resize(it.second->shape.size());
      for(size_t i = 0; i < it.second->shape.size(); ++i)
        shape.set(i, (size_t)it.second->shape[i]);
    }
    Item item;
    item.name = it.first;
    item.shape = shape;
    item.bytes.swap(it.second->bytes);
    
    if (alham_map.find(item.name) != alham_map.end()) {
      LOG(info, "Deq: {}  {}, {}, {}", it.first, shape[0], shape[1], item.bytes.size());

      int shift = shape[0] * shape[1];
      std::vector<float> newbytes(shift * 2, 0);
    
      for (int i = 0;i < item.bytes.size();i++){
        uint8_t c = item.bytes[i];
        newbytes[i] = toFloat(c >> 4, alham_map[item.name]);
        newbytes[i + shift] = toFloat(c & 0xf, alham_map[item.name]);
        if (i == 0) LOG(info, "test {} -> {} {} |  val {} ", (uint8_t) c,  (uint8_t) c >> 4, (uint8_t) c & 0xf, newbytes[i]);
        
      }
      item.shape.set(0, shape[0] * 2);
      char* c = (char*) newbytes.data();
      std::vector<char> x(c, c + shift * 8);
      item.bytes = x;
    } 
    items.emplace_back(std::move(item));
  }
  LOG(info, "DONE ALL");
}

std::vector<Item> loadItems(const std::string& fileName) {
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

std::vector<Item> loadItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, false);
  return items;
}

std::vector<Item> mmapItems(const void* ptr) {
  std::vector<Item> items;
  binary::loadItems(ptr, items, true);
  return items;
}

// @TODO: make cnpy and our wrapper talk to each other in terms of types
// or implement our own saving routines for npz based on npy, probably better.
void saveItemsNpz(const std::string& fileName, const std::vector<Item>& items) {
  std::vector<cnpy::NpzItem> npzItems;
  for(auto& item : items) {
    std::vector<unsigned int> shape(item.shape.begin(), item.shape.end());
    char type;

    if(item.type == Type::float32)
      type = cnpy::map_type(typeid(float));
    else if(item.type == Type::float64)
      type = cnpy::map_type(typeid(double));
    else if(item.type == Type::int8)
      type = cnpy::map_type(typeid(char));
    else
      ABORT("Other types not supported yet");

    npzItems.emplace_back(
        item.name, item.bytes, shape, type, sizeOf(item.type));
  }
  cnpy::npz_save(fileName, npzItems);
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
