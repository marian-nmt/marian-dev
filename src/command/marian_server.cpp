#include "translator/server_common.h"
#include "translator/translator.h"

int main(int argc, char **argv) {
  using namespace marian;

  auto options = parseOptions(argc, argv, cli::mode::server, true);
  auto task = New<TranslateService<BeamSearch>>(options);

  return runServer(task, options);
}
