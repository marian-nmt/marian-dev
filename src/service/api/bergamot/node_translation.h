#pragma once
#include "service/common/translation_service.h"
#include "3rd_party/rapidjson/include/rapidjson/document.h"

namespace marian {
namespace server {

class NodeTranslation {
public:
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
private:
  OutputOptions reportingOptions_; // what to return
  TranslationOptions translationOptions_; // how to translate
  TranslationService& service_;

  std::vector<NodeTranslation> children_;
  rapidjson::Value* node_;
  Ptr<PlainTextTranslation> translation_;
  bool ends_with_eol_char_{false};
  splitmode smode_{splitmode::wrapped_text};

  void setOptions(rapidjson::Value* n,
                  NodeTranslation const* parent,
                  const std::string& options_field);
 public:
  NodeTranslation(rapidjson::Value* n,
                  TranslationService& service,
                  std::string payload_field="text",
                  std::string options_field="options",
                  NodeTranslation* parent=NULL);

  void finish(rapidjson::Document::AllocatorType& alloc);

};

}} // end of namespace marian::server
