#include <iostream>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "marian.h"
#include "functional/functional.h"

int main(int argc, char** argv) {

using namespace marian;
using namespace marian::functional;

var<1> x;
var<2> y;
var<3> z;

auto f = x * logit(x);
auto df_x = simplify(grad(f, x));

auto df_y = simplify(grad(f, y));
auto df_z = simplify(grad(f, z));

std::cerr << f.to_string() << std::endl;

std::cerr << df_x.to_string() << std::endl;
std::cerr << df_y.to_string() << std::endl;
std::cerr << df_z.to_string() << std::endl;

}
