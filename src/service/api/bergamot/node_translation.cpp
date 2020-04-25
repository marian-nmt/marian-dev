#include "node_translation.h"

namespace marian {
namespace server {

NodeTranslation::
NodeTranslation(rapidjson::Value* n,
                TranslationService& service,
                std::string payload_field="text",
                std::string options_field="options",
                NodeTranslation* parent=NULL)
  : service_(service), node_(n) {
  if (n == NULL)
    return; // nothing to do
  setOptions(n, parent, options_field);

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
    if (details_) translationOptions_.nbest = get(*details_,"NBest",1);
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
    reportingOptions_ = parent_->reportingOptions;
    translationOptions_ = parent_->translationOptions_;
  }

  // Look for local overrrides:
  if (n->IsObject()){
    auto x = n->FindMember(options_field.c_str());
    if (x != n->MemberEnd() && x->value.IsObject()){
      const auto& v = x->value;
      if (v.HasMember("InputFormat")) {
        smode_ = interpret_splitmode(v["InputFormat"].GetString());
      }
      setOptions(reportingOptions_, v);
      translationOptions_.nbest = get(v, "NBest", translationOptions_.nbest);
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

  if (reportingOptions_.noDetail()) {
    std::string T = translation_->toString();
    node_.SetString(T.c_str(), T.size(), alloc);
  }
  else {
    node_.SetArray(); // list of paragraphs
    Value* current_paragraph = NULL;
    for (size_t i = 0; i < translation_->size(); ++i) {
      auto j = translation_->await(i);
      if (j->nbest.size() == 0) {
        current_paragraph = NULL;
      }
      else {
        if (current_paragraph == NULL) { // => start a new paragraph
          node_.PushBack(Value(kArrayType).Move(), alloc);
          current_paragraph = &(node_[node_.Size() - 1]);
        }
        auto s = job2json(*j, service, reportingOptions_, alloc);
        current_paragraph->PushBack(s.Move(), alloc);
      }
    }
  }
}


}} // end of namespace marian::server
