#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

using namespace std;
using namespace mlpack;
// NeighborSearch and NearestNeighborSort
using namespace mlpack::neighbor;
// ManhattanDistance
using namespace mlpack::metric;

void mlModel()
{

	arma::mat data;

	NeighborSearch<NearestNeighborSort, ManhattanDistance> nn(data);


	arma::Mat<size_t> neighbors; // Matrices to hold
	arma::mat distances; // the results

	
	nn.Search(1, neighbors, distances);


	// Print out each neighbor and its distance.
	for (size_t i = 0; i < neighbors.n_elem; ++i)
	{
		std::cout << "Nearest neighbor of point " << i << " is point "
				<< neighbors[i] << " and the distance is "
				<< distances[i] << ".\n";
	}
}

int main()
{
	mlModel();
	return 0;
}
