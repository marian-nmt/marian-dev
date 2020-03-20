// -*- mode: c++; indent-tabs-mode: nil; tab-width: 2 -*-
#pragma once
#include "translation_service.h"
#include "crow.h"

namespace marian {
namespace server {

template<class Search=BeamSearch>
class JsonRequestHandler {
  Ptr<TranslationService<Search>> service_;
public:
  JsonRequestHandler(Ptr<TranslationService<Search>> service)
    : service_(service) { }

  virtual Ptr<crow::json:wvalue> operator()(std::string request) = 0;
};

}}
