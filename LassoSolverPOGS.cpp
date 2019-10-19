#include <random>
#include <vector>

#include "pogs.h"

int main() {
  // Generate Data
  size_t m = 100, n = 10;
  std::vector<double> A(m * n);
  std::vector<double> b(m);
  std::vector<double> x(n);
  std::vector<double> y(m);

  std::default_random_engine generator;
  std::normal_distribution<double> n_dist(0.0, 1.0);

  for (unsigned int i = 0; i < m * n; ++i)
    A[i] = n_dist(generator);

  for (unsigned int i = 0; i < m; ++i)
    b[i] = n_dist(generator);

  // Populate f and g
  PogsData<double, double*> pogs_data(A.data(), m, n);
  pogs_data.x = x.data();
  pogs_data.y = y.data();

  pogs_data.f.reserve(m);
  for (unsigned int i = 0; i < m; ++i)
    pogs_data.f.emplace_back(kSquare, 1.0, b[i]);

  pogs_data.g.reserve(n);
  for (unsigned int i = 0; i < n; ++i)
    pogs_data.g.emplace_back(kAbs, 0.5);

  // Solve
  Pogs(&pogs_data);
}
