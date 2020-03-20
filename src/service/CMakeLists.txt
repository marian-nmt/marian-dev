# link_directories(..)
message("BOOST COMPONENTS ${BOOST_COMPONENTS}")
# Dependencies. Apparently paths need to be set before targets are
# created, so we ensure dependencies first, then create targets.
if (COMPILE_REST_SERVER OR COMPILE_AMQP_WORKER OR COMPILE_WEBSOCKET_SERVER)
  find_package(PCRECPP) # needed for ssplit-cpp
  add_library(marian-service STATIC
    common/queued_input.cpp
    common/translation_job.cpp
    api/rapidjson_utils.cpp
    )
  target_compile_options(marian-service PUBLIC ${ALL_WARNINGS})
endif(COMPILE_REST_SERVER OR COMPILE_AMQP_WORKER OR COMPILE_WEBSOCKET_SERVER)

if (COMPILE_REST_SERVER)
  add_executable(rest-server rest/rest_server.cpp)
  set_target_properties(rest-server PROPERTIES OUTPUT marian-rest-server)
  target_compile_options(rest-server PUBLIC ${ALL_WARNINGS})
  set(EXECUTABLES ${EXECUTABLES} rest-server)
  # TODO: use of static vs shared pcre and pcrecpp libraries should
  # depend on USE_STATIC_LIBS, but I couldn't get that to work.
  target_link_libraries(rest-server marian ${EXT_LIBS} marian-service ssplit pcrecpp.a pcre.a)
endif(COMPILE_REST_SERVER)

if (COMPILE_AMQP_WORKER)
  add_executable(amqp-worker amqp/amqp_worker.cpp)
  set_target_properties(amqp-worker PROPERTIES OUTPUT marian-amqp-worker)
  target_compile_options(amqp-worker PUBLIC ${ALL_WARNINGS})
  set(EXECUTABLES ${EXECUTABLES} amqp-worker)
  # TODO: use of static vs shared pcre and pcrecpp libraries should
  # depend on USE_STATIC_LIBS, but I couldn't get that to work.
  target_link_libraries(amqp-worker marian ${EXT_LIBS} marian-service amqpcpp ev ssplit pcrecpp.a pcre.a)
endif(COMPILE_AMQP_WORKER)

if (COMPILE_WEBSOCKET_SERVER)
  add_executable(websocket-server server/websocket-server.cpp)
  set_target_properties(websocket-server PROPERTIES OUTPUT websocket-server)
  target_compile_options(websocket-server PUBLIC ${ALL_WARNINGS})
  # TODO: use of static vs shared pcre and pcrecpp libraries should
  # depend on USE_STATIC_LIBS, but I couldn't get that to work.
  target_link_libraries(websocket-server marian ${EXT_LIBS} marian-service ssplit pcrecpp.a pcre.a)
  set(EXECUTABLES ${EXECUTABLES} websocket-server)
endif(COMPILE_WEBSOCKET_SERVER)
