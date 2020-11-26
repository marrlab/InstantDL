#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>

using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	    const T * medge_weightArray;
	public:
		AffinityGraphCompare(const T * edge_weightArray){
			medge_weightArray = edge_weightArray;
		}
		bool operator() (const int& ind1, const int& ind2) const {
			return (medge_weightArray[ind1] > medge_weightArray[ind2]);
		}
};

inline bool fiszero(float x)
{
  const float epsilon =  1e-5;
  return std::abs(x) <= epsilon;
};

/*
 * Compute the MALIS loss function and its derivative wrt the affinity graph
 * MAXIMUM spanning tree
 * Author: Srini Turaga (sturaga@mit.edu)
 * All rights reserved
 */
void malis_loss_weights_cpp(const int n_vert, const int* seg,
               const int n_edge, const int* node1, const int* node2, const float* edge_weight,
               const int pos,
               int* counts){


    /* Disjoint sets and sparse overlap vectors */
    vector<map<int,int> > overlap(n_vert);
    vector<int> rank(n_vert);
    vector<int> parent(n_vert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<n_vert; ++i){
        dsets.make_set(i);
        if (0!=seg[i]) {
            overlap[i].insert(pair<int,int>(seg[i],1)); // (ID in GT, count of paths)

        }
    }


    /* Sort all the edges in increasing order of weight */
    std::vector< int > pqueue( n_edge );
    int edge_count = 0;
    for ( int i = 0; i < n_edge; i++ ){
        if ((node1[i]>=0) && (node1[i]<n_vert) && (node2[i]>=0) && (node2[i]<n_vert))
	        pqueue[ edge_count++ ] = i;
    }

    pqueue.resize(edge_count);
    stable_sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edge_weight ) );
//    for ( int i = 0; i < j; i++ )
//       cout<<"i="<<i<<" edge="<<pqueue[i]<<" weight="<<edge_weight[pqueue[i]]<<endl;


    /* Start MaxST */
    int min_edge;
    int set1, set2;
    int n_pairs = 0;
    map<int,int>::iterator it1, it2;

    /* Start Kruskal's */
    //cout<<"START Kruskal"<<endl;
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {
        min_edge = pqueue[i];
         
        set1 = dsets.find_set(node1[min_edge]);
        set2 = dsets.find_set(node2[min_edge]);
        //cout<<"i="<<i<<" node1="<<node1[min_edge]<<" node2="<<node2[min_edge]<<" edge="<<pqueue[i]<<" weight="<<edge_weight[pqueue[i]]<<" set1="<<set1<<" set2="<<set2<<endl;

        if (set1!=set2){
            //cout<<"  JOIN set "<<set1<<" and set "<<set2<<" with edge "<<min_edge<<" at weight "<<edge_weight[min_edge]<<endl;
            dsets.link(set1, set2);

            /* compute the number of pairs merged by this MST edge */
            for (it1 = overlap[set1].begin(); it1 != overlap[set1].end(); ++it1) { // loop over the overlap dict for node1 of min edge (set1)
                for (it2 = overlap[set2].begin(); it2 != overlap[set2].end(); ++it2) {// loop over the overlap dict for node2 of min edge (set2)

                    n_pairs = it1->second * it2->second;
                    //cout<<"    Iterating over: "<<" (ID="<<it1->first<<") and(ID="<<it2->first<<")"<<endl;
                    if (pos && (it1->first == it2->first)) { // these nodes have same ID in GT / are connected
                        //cout<<"    For min_edge "<<min_edge<<" linking "<<set1<<" (ID="<<it1->first<<") and "<<set2<<"(ID="<<it2->first<<") - adding N="<<n_pairs<<endl;
                        //cout<<"    For min_edge "<<min_edge<<" between "<<node1[min_edge]<<" (ID="<<it1->first<<") and "<<node2[min_edge]<<"(ID="<<it2->first<<") - adding N="<<n_pairs<<endl;
                        counts[min_edge] += n_pairs;
                    } else if ((!pos) && (it1->first != it2->first)) { // these nodes have different ID in GT / are disconnected
                        //cout<<"    For min_edge "<<min_edge<<" linking "<<set1<<" (ID="<<it1->first<<") and "<<set2<<"(ID="<<it2->first<<") - adding N="<<n_pairs<<endl;
                        counts[min_edge] += n_pairs;
                    }
                }
            }

            /* move the pixel bags of the non-representative to the representative */
            if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
                swap(set1,set2);

            it2 = overlap[set2].begin();
            while (it2 != overlap[set2].end()) {
                it1 = overlap[set1].find(it2->first);
                if (it1 == overlap[set1].end()) {
                    overlap[set1].insert(pair<int,int>(it2->first,it2->second));
                } else {
                    it1->second += it2->second;
                }
                overlap[set2].erase(it2++);
            }
        } // end link

    } // end while
}


void connected_components_cpp(const int n_vert,
               const int n_edge, const int* node1, const int* node2, const float* edge_weight, const int size_thresh,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(n_vert);
    vector<int> parent(n_vert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<n_vert; ++i)
        dsets.make_set(i);

    /* union */
    for (int i = 0; i < n_edge; ++i )
        // check bounds to make sure the nodes are valid
        if ((!fiszero(edge_weight[i])) && // check connectivity
            (node1[i]>=0) &&
            (node1[i]<n_vert) &&
            (node2[i]>=0) &&
            (node2[i]<n_vert))
            dsets.union_set(node1[i],node2[i]);

    /* find */
    for (int i = 0; i < n_vert; ++i)
        seg[i] = dsets.find_set(i)+1;

    /* count elements per component */
    map<int,int> comp_sizes;
    for (int i = 0; i < n_vert; ++i)
       ++comp_sizes[seg[i]];

    /* set small elements to 0 */
    for (int i = 0; i < n_vert; ++i)
        if (comp_sizes[seg[i]]<=size_thresh)
        seg[i] = 0;  


}



void marker_watershed_cpp(const int n_vert, const int* marker,
               const int n_edge, const int* node1, const int* node2, const float* edge_weight, const int size_thresh,
               int* seg){

    /* Make disjoint sets */
    vector<int> rank(n_vert);
    vector<int> parent(n_vert);
    boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
    for (int i=0; i<n_vert; ++i)
        dsets.make_set(i);

    /* initialize output array and find representatives of each class */
    std::map<int,int> components;
    for (int i=0; i<n_vert; ++i){
        seg[i] = marker[i];
        if (seg[i] > 0)
            components[seg[i]] = i;
    }

    // merge vertices labeled with the same marker
    for (int i=0; i<n_vert; ++i)
        if (seg[i] > 0)
            dsets.union_set(components[seg[i]],i);

    /* Sort all the edges in decreasing order of weight */
    std::vector<int> pqueue( n_edge );
    int j = 0;
    for (int i = 0; i < n_edge; ++i)
        if ((!fiszero(edge_weight[i])) &&
            (node1[i]>=0) && (node1[i]<n_vert) &&
            (node2[i]>=0) && (node2[i]<n_vert) &&
            (marker[node1[i]]>=0) && (marker[node2[i]]>=0))
                pqueue[ j++ ] = i;
    unsigned long nValidEdge = j;
    pqueue.resize(nValidEdge);
    sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( edge_weight ) );

    /* Start MST */
    int set1, set2, label_of_set1, label_of_set2;
    for (unsigned int i = 0; i < pqueue.size(); ++i ) {

        set1=dsets.find_set(node1[i]);
        set2=dsets.find_set(node2[i]);
        label_of_set1 = seg[set1];
        label_of_set2 = seg[set2];

        if ((set1!=set2) &&
            ( ((label_of_set1==0) && (marker[set1]==0)) ||
             ((label_of_set2==0) && (marker[set1]==0))) ){

            dsets.link(set1, set2);
            // either label_of_set1 is 0 or label_of_set2 is 0.
            seg[dsets.find_set(set1)] = std::max(label_of_set1,label_of_set2);
            
        }

    }

    // write out the final coloring
    for (int i=0; i<n_vert; i++)
        seg[i] = seg[dsets.find_set(i)];


    /* count elements per component */
    map<int,int> comp_sizes;
    for (int i = 0; i < n_vert; ++i)
       ++comp_sizes[seg[i]];

    /* set small elements to 0 */
    for (int i = 0; i < n_vert; ++i)
        if (comp_sizes[seg[i]]<=size_thresh)
        seg[i] = 0;  

}

