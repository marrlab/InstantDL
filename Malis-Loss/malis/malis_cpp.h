#ifndef MALIS_H
#define MALIS_H

void connected_components_cpp(const int n_vert,
                              const int n_edge,
                              const int* node1,
                              const int* node2,
                              const float* edge_weight,
                              const int size_thresh,
                              int* seg);

void malis_loss_weights_cpp(const int n_vert,
                            const int* seg,
                            const int n_edge,
                            const int* node1,
                            const int* node2,
                            const float* edge_weight,
                            const int pos,
                            int* counts);

void marker_watershed_cpp(const int n_vert,
                          const int* marker,
                          const int n_edge,
                          const int* node1,
                          const int* node2,
                          const float* edge_weight,
                          const int size_thresh,
                          int* seg);

#endif