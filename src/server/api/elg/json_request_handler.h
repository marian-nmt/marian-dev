#pragma once
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
    auto n = get(&request, {"request","params","inputFormat"});
    splitmode smode = splitmode::wrapped_text;
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

  void apiError(rapidjson::Document& D, std::string const errmsg) const {
    auto& alloc = D.GetAllocator();
    rapidjson::Value& n = D;
    auto e = rapidjson::ensure_path(n, alloc, "failure", "errors");
    e->SetArray().PushBack(rapidjson::Value(errmsg.c_str(),alloc).Move(), alloc);
  }

public:
  typedef ug::ssplit::SentenceStream::splitmode splitmode;
  ElgJsonRequestHandlerV1(Service& service)
    : JsonRequestHandlerBaseClass<Service>(service) { }



  Ptr<rapidjson::Document>
  operator()(char const* body) const {
    Ptr<rapidjson::Document> D(new rapidjson::Document());
    D->Parse(body);
    if (!D->IsObject()) {
      apiError(*D, "Invalid Json");
      LOG(debug, "INVALID JSON: {}", body);
      return D;
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
    if (request.HasMember("metadata")){
      D->AddMember("metadata", {}, alloc);
      (*D)["metadata"].CopyFrom(request["metadata"], alloc);
    }

    // get translation parameters; currently, the only parameter
    // considered is the sentence splitting mode:
    // one sentence per line, one paragraph per line, or wrapped text
    // (with emptly lines demarcating paragraph boundaries)
    auto smode = getSentenceSplitMode(request);

    // get the actual payload
    auto payload = get(&request, {"request", "content"});
    if (!payload){
      apiError(*D, "No content to translate provided.");
    }
    if (!payload->IsString()){
      apiError(*D, "Translation payload ('request|content') is not a string.");
    }
    std::string translation
      = this->service_.translate(payload->GetString(),smode)->await();

    auto& r = D->AddMember("response",{},alloc)["response"].SetObject();
    r.AddMember("type", "texts", alloc);
    rapidjson::Value x(rapidjson::kObjectType);
    x.AddMember("text", {}, alloc)["text"]
      .SetString(translation.c_str(), translation.size(), alloc);
    r.AddMember("texts", {}, alloc)["texts"]
      .SetArray().PushBack(x.Move(),alloc);
    return D;
  }
};
}} // end of namespace marian::server::elg
