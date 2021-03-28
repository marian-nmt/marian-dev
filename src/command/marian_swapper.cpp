#include "translator/swappable.h"
#include <iostream>

/* Demo program: run with options for any of the models */
int main(int argc, char** argv) {
  using namespace marian;
  Ptr<Options> options = parseOptions(argc, argv, cli::mode::translation);
  SwappableSlot slot(options);
  SwappableModel pten(options,
      "/home/ubuntu/consistent-big-models/padded/pten.npz",
      {"/home/ubuntu/consistent-big-models/padded/pten.vocab"},
      "/home/ubuntu/consistent-big-models/padded/pten.vocab");

  SwappableModel enit(options,
      "/home/ubuntu/consistent-big-models/padded/enit.npz",
      {"/home/ubuntu/consistent-big-models/padded/enit.vocab"},
      "/home/ubuntu/consistent-big-models/padded/enit.vocab");

  const SwappableModel *model = &pten;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (line == " TRANSLATE PTEN") {
      model = &pten;
    } else if (line == " TRANSLATE ENIT") {
      model = &enit;
    } else {
      slot.Translate(*model, {line});
    }
  }

  return 0;
}
