#include "node_translation.h"
#include "service/api/rapidjson_utils.h"
#include "service/common/translation_service.h"
#include "service/common/plaintext_translation.h"
#include <string>

// After some thought and investivation, I decided to
// go with camelCase for propertNames.
// see here: https://github.com/json-api/json-api/issues/1255

namespace marian {
namespace server {

NodeTranslation::
NodeTranslation(rapidjson::Value* n,
                TranslationService& service,
                std::string payload_field/*="text"*/,
                std::string options_field/*="options"*/,
                NodeTranslation* parent/*=NULL*/)
  : service_(service), node_(n) {
  if (n == NULL)
    return; // nothing to do
  setOptions(n, parent, options_field);
  //@TODO: check if service provides word alignments
  //       add error if not but it is requested.
  //@TOOD: check for source sentence length, add warning if
  //       cropped.

  if (n->IsObject()){
    auto x = n->FindMember(payload_field.c_str());
    if (x != n->MemberEnd()){
      auto c = NodeTranslation(&(x->value),
                               service,
                               payload_field,
                               options_field,
                               this);
      children_.push_back(std::move(c));
    }
  }
  else if (n->IsString()) {
    translation_.reset(new PlainTextTranslation(n->GetString(),
                                                service,
                                                translationOptions_,
                                                smode_));
  }
  else if (n->IsArray()) {
    for (auto c = n->Begin(); c != n->End(); ++c){
      auto x = NodeTranslation(c, service, payload_field, options_field, this);
      children_.push_back(std::move(x));
    }
  }
}

void
NodeTranslation::
setOptions(rapidjson::Value* n,
           NodeTranslation const* parent,
           const std::string& options_field){
  // Inherit parameters from parent:
  if (parent) {
    smode_ = parent->smode_;
    reportingOptions_ = parent->reportingOptions_;
    translationOptions_ = parent->translationOptions_;
  }

  // Look for local overrrides:
  if (n->IsObject()){
    auto x = n->FindMember(options_field.c_str());
    if (x != n->MemberEnd() && x->value.IsObject()){
      const auto& v = x->value;
      if (v.HasMember("inputFormat")) {
        smode_ =string2splitmode(v["inputFormat"].GetString());
      }
      rapidjson::setOptions(reportingOptions_, v);
      translationOptions_.nbest = get(v, "nBest", translationOptions_.nbest);
    }
  }
}

void
NodeTranslation::
finish(rapidjson::Document::AllocatorType& alloc) {
  using rapidjson::Value;
  using rapidjson::kArrayType;
  using rapidjson::kObjectType;
  for (auto& c: children_) {
    c.finish(alloc);
  }

  if (!translation_)
    return;

  if (reportingOptions_.noDetails()) {
    std::string T = translation_->toString();
    node_->SetString(T.c_str(), T.size(), alloc);
  }
  else {
    node_->SetArray(); // list of paragraphs
    Value* current_paragraph = NULL;
    for (size_t i = 0; i < translation_->size(); ++i) {
      auto j = translation_->await(i);
      if (j->nbest.size() == 0) {
        current_paragraph = NULL;
      }
      else {
        if (current_paragraph == NULL) { // => start a new paragraph
          node_->PushBack(Value(kArrayType).Move(), alloc);
          current_paragraph = &((*node_)[node_->Size() - 1]);
        }
        auto s = job2json(*j, service_, reportingOptions_, alloc);
        current_paragraph->PushBack(s.Move(), alloc);
      }
    }
  }
}


}} // end of namespace marian::server
