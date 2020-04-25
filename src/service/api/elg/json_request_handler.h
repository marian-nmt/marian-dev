#pragma once
#include "service/common/plaintext_translation.h"
// Do not include this file directly. It is included by ../json_request_handler.h
namespace marian {
namespace server {

template<class Service>
class ElgJsonRequestHandlerV1
  : public JsonRequestHandlerBaseClass<Service>{

  // Return pointer to rapidjson::Value if path exists, NULL otherwise
  rapidjson::Value const*
  get(rapidjson::Value const* node, std::vector<char const*> path) const {
    for (char const* f: path) {
      if (!node->IsObject())
        return NULL;
      auto m = node->FindMember(f);
      if (m == node->MemberEnd())
        return NULL;
      node = &(m->value);
    }
    return node;
  }

  ug::ssplit::SentenceStream::splitmode
  getSentenceSplitMode(rapidjson::Value const& request) const {
    auto n = get(&request, {"params","inputFormat"});
    // splitmode smode = splitmode::wrapped_text;
    if (n && n->IsString())
      {
        std::string input_format = n->GetString();
        if (input_format == "sentence")
          return splitmode::one_sentence_per_line;
        else if (input_format == "paragraph")
          return splitmode::one_paragraph_per_line;
      }
    return splitmode::wrapped_text;
  }

  class NodeWrapper {
    typedef PlainTextTranslation tjob;
    std::vector<NodeWrapper> children_;
    Ptr<tjob> translation_;
  public:
    NodeWrapper(rapidjson::Value const& n,
                Service& service,
                TranslationOptions& topts,
                ug::ssplit::SentenceStream::splitmode const& smode) {
      if (n.IsObject()) {
        auto x = n.FindMember("content");
        if (x != n.MemberEnd() && x->value.IsString()) {
          std::string input = x->value.GetString();
          translation_.reset(new tjob(input, service, topts, smode));
        }
        auto y = n.FindMember("texts");
        if (y != n.MemberEnd() && y->value.IsArray()) {
          for (auto c = y->value.Begin(); c != y->value.End(); ++c) {
            auto z = NodeWrapper(*c, service, topts, smode);
            children_.push_back(std::move(z));
          }
        }
      }
    }

    void finish(rapidjson::Value& n, rapidjson::Document::AllocatorType& alloc) {
      rapidjson::Value x(rapidjson::kObjectType);
      if (translation_) {
        std::string t = translation_->toString();
        x.AddMember("content", {}, alloc)["content"].SetString(t.c_str(), t.size(), alloc);
        // x.AddMember("content", translation_->await(), alloc);
      }
      if (children_.size()) {
        auto& a = x.AddMember("texts",{},alloc)["texts"].SetArray();
        for (auto& c: children_) {
          c.finish(a, alloc);
        }
      }
      n.PushBack(x.Move(),alloc);
    }
  };

  enum class ElgErrCode { internal_server_error,
                          invalid_request,
                          missing_request,
                          unsupported_request_type,
                          unsupported_mime_type,
                          request_too_large };

  // We use a function below instead of static members
  // because of the move semantics of the assignment operator
  // in rapidjson. See rapidjson documentation.
  rapidjson::Value api_error(ElgErrCode errCode, rapidjson::Document::AllocatorType& alloc) const {
    rapidjson::Value n(rapidjson::kObjectType);
    if (errCode == ElgErrCode::invalid_request){
      n.AddMember("code","elg.request.invalid", alloc);
      n.AddMember("text","Invalid request message.", alloc);
    }
    else if (errCode == ElgErrCode::missing_request){
      n.AddMember("code","elg.request.missing", alloc);
      n.AddMember("text","No request provided in message.", alloc);
    }
    else if (errCode == ElgErrCode::unsupported_request_type){
      n.AddMember("code","elg.request.type.unsupported", alloc);
      n.AddMember("text","Request type {0} not supported by this service.", alloc);
    }
    else if (errCode == ElgErrCode::request_too_large){
      n.AddMember("code","elg.request.too.large", alloc);
      n.AddMember("text","Request size too large.", alloc);
    }
    else if (errCode == ElgErrCode::unsupported_mime_type){
      n.AddMember("code","elg.request.text.mimeType.unsupported", alloc);
      n.AddMember("text","MIME type {0} not supported by this service.", alloc);
    }
    else {
      n.AddMember("code","elg.service.internalError", alloc);
      n.AddMember("text","Internal error during processing: {0}", alloc);
    }
    return n;
  }


public:
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
  ElgJsonRequestHandlerV1(Service& service)
    : JsonRequestHandlerBaseClass<Service>(service) {
  }

  Ptr<rapidjson::Document>
  operator()(char const* body) const {
    Ptr<rapidjson::Document> D(new rapidjson::Document());
    D->Parse(body);
    if (!D->IsObject()) {
      Ptr<rapidjson::Document> R(new rapidjson::Document());
      auto& alloc = R->GetAllocator();
      auto e = rapidjson::ensure_path(R->SetObject(), alloc, "failure", "errors");
      auto n = api_error(ElgErrCode::invalid_request, alloc);
      e->SetArray().PushBack(n, alloc);
      return R;
    }
    LOG(debug, "PARSED: {}", serialize(*D));
    return (*this)(*D);
  }

  Ptr<rapidjson::Document>
  operator()(std::string const& body) const override {
    return (*this)(body.c_str());
  }

  Ptr<rapidjson::Document>
  operator()(rapidjson::Value const& request) const {

    // create a response JSON document
    auto D = std::make_shared<rapidjson::Document>();
    auto& alloc = D->GetAllocator();
    D->SetObject();

    // Copy metadata from request.
    if (request.HasMember("metaData")){
      D->AddMember("metaData", {}, alloc);
      (*D)["metaData"].CopyFrom(request["metaData"], alloc);
    }

    if (!request.HasMember("content") && !request.HasMember("texts")){
      auto e = rapidjson::ensure_path(*D, alloc, "failure", "errors");
      auto n = api_error(ElgErrCode::missing_request, D->GetAllocator());
      e->SetArray().PushBack(n, D->GetAllocator());
      return D;
    }

    // get translation parameters; currently, the only parameter
    // considered is the sentence splitting mode:
    // one sentence per line, one paragraph per line, or wrapped text
    // (with emptly lines demarcating paragraph boundaries)
    auto smode = getSentenceSplitMode(request);

    // Perform translation. NodeWrappers constructor recurses through
    // the request's possibly hierarchical structure and pushes all
    // chunks that need to be translated into the service's input queue.
    // Each node has a std::future to the eventual translation.
    // NodeWrapper::finish recurses through the structure and adds the
    // translations to the response document D.
    auto r = rapidjson::ensure_path(*D, alloc, "response", "texts");
    rapidjson::Value n(rapidjson::kArrayType);
    TranslationOptions topts;
    auto p = get(&request, {"params", "NBest"});
    if (p && p->IsObject())
      topts.nbest = p->GetInt();
    NodeWrapper(request, this->service, topts, smode).finish(n, alloc);
    *r = n;
    (*D)["response"].AddMember("type","texts",alloc);
    return D;
  }
};
}} // end of namespace marian::server::elg
