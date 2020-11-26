#pragma once

#include "types.hpp"

#include <iostream>

/**
 * Perform a watershed segmentation on an affinity graph.
 *
 * @param aff [in]
 *              A multi-array holding the affinity graph with shape 
 *              (3,depth,height,width).
 * @param low [in]
 * @param high [in]
 * @param seg [out]
 *              A reference to a segmentation multi-array that will be used to 
 *              store the segmentation. The caller has to ensure it is of the 
 *              correct shape (depth,height,width).
 * @param counts [out]
 *              A reference to a counts_t data structure that will be used to 
 *              store the sizes of the found regions.
 */
template<typename AG, typename V>
inline
void
watershed(
        const AG& aff,
        typename AG::element low,
        typename AG::element high,
        V& seg,
        counts_t<std::size_t>& counts)
{
    typedef typename AG::element F;
    typedef typename V::element  ID;

    using traits = watershed_traits<ID>;

    std::ptrdiff_t zdim = aff.shape()[1];
    std::ptrdiff_t ydim = aff.shape()[2];
    std::ptrdiff_t xdim = aff.shape()[3];

    std::ptrdiff_t size = xdim * ydim * zdim;

    assert(seg.shape()[0] == zdim);
    assert(seg.shape()[1] == ydim);
    assert(seg.shape()[2] == xdim);

    counts.resize(1);
    counts[0] = 0;

    ID* seg_raw = seg.data();

    for ( std::ptrdiff_t z = 0; z < zdim; ++z )
        for ( std::ptrdiff_t y = 0; y < ydim; ++y )
            for ( std::ptrdiff_t x = 0; x < xdim; ++x )
            {
                ID& id = seg[z][y][x] = 0;

                F negz = (z>0) ? aff[0][z][y][x] : low;
                F negy = (y>0) ? aff[1][z][y][x] : low;
                F negx = (x>0) ? aff[2][z][y][x] : low;
                F posz = (z<(zdim-1)) ? aff[0][z+1][y][x] : low;
                F posy = (y<(ydim-1)) ? aff[1][z][y+1][x] : low;
                F posx = (x<(xdim-1)) ? aff[2][z][y][x+1] : low;

                F m = std::max({negx,negy,negz,posx,posy,posz});

                if ( m > low )
                {
                    if ( negz == m || negz >= high ) { id |= 0x01; }
                    if ( negy == m || negy >= high ) { id |= 0x02; }
                    if ( negx == m || negx >= high ) { id |= 0x04; }
                    if ( posz == m || posz >= high ) { id |= 0x08; }
                    if ( posy == m || posy >= high ) { id |= 0x10; }
                    if ( posx == m || posx >= high ) { id |= 0x20; }
                }
            }


    //                              -z          -y     -x  +z         +y    +x
    const std::ptrdiff_t dir[6] = { -ydim*xdim, -xdim, -1, ydim*xdim, xdim, 1 };
    const ID dirmask[6]  = { 0x01, 0x02, 0x04, 0x08, 0x10, 0x20 };
    const ID idirmask[6] = { 0x08, 0x10, 0x20, 0x01, 0x02, 0x04 };

    // get plato corners

    std::vector<std::ptrdiff_t> bfs;

    for ( std::ptrdiff_t idx = 0; idx < size; ++idx )
    {
        for ( std::ptrdiff_t d = 0; d < 6; ++d )
        {
            if ( seg_raw[idx] & dirmask[d] )
            {
                if ( !(seg_raw[idx+dir[d]] & idirmask[d]) )
                {
                    seg_raw[idx] |= 0x40;
                    bfs.push_back(idx);
                    d = 6; // break;
                }
            }
        }
    }

    // divide the plateaus

    std::size_t bfs_index = 0;

    while ( bfs_index < bfs.size() )
    {
        std::ptrdiff_t idx = bfs[bfs_index];

        ID to_set = 0;

        for ( std::ptrdiff_t d = 0; d < 6; ++d )
        {
            if ( seg_raw[idx] & dirmask[d] )
            {
                if ( seg_raw[idx+dir[d]] & idirmask[d] )
                {
                    if ( !( seg_raw[idx+dir[d]] & 0x40 ) )
                    {
                        bfs.push_back(idx+dir[d]);
                        seg_raw[idx+dir[d]] |= 0x40;
                    }
                }
                else
                {
                    to_set = dirmask[d];
                }
            }
        }
        seg_raw[idx] = to_set;
        ++bfs_index;
    }

    bfs.clear();

    // main watershed logic

    ID next_id = 1;

    for ( std::ptrdiff_t idx = 0; idx < size; ++idx )
    {
        if ( seg_raw[idx] == 0 )
        {
            seg_raw[idx] |= traits::high_bit;
            ++counts[0];
        }

        if ( !( seg_raw[idx] & traits::high_bit ) && seg_raw[idx] )
        {
            bfs.push_back(idx);
            bfs_index = 0;
            seg_raw[idx] |= 0x40;

            while ( bfs_index < bfs.size() )
            {
                std::ptrdiff_t me = bfs[bfs_index];

                for ( std::ptrdiff_t d = 0; d < 6; ++d )
                {
                    if ( seg_raw[me] & dirmask[d] )
                    {
                        std::ptrdiff_t him = me + dir[d];
                        if ( seg_raw[him] & traits::high_bit )
                        {
                            counts[ seg_raw[him] & ~traits::high_bit ]
                                += bfs.size();

                            for ( auto& it: bfs )
                            {
                                seg_raw[it] = seg_raw[him];
                            }

                            bfs.clear();
                            d = 6; // break
                        }
                        else if ( !( seg_raw[him] & 0x40 ) )
                        {
                            seg_raw[him] |= 0x40;
                            bfs.push_back( him );

                        }
                    }
                }
                ++bfs_index;
            }

            if ( bfs.size() )
            {
                counts.push_back( bfs.size() );
                for ( auto& it: bfs )
                {
                    seg_raw[it] = traits::high_bit | next_id;
                }
                ++next_id;
                bfs.clear();
            }
        }
    }

    std::cout << "found: " << (next_id-1) << " components\n";

    for ( std::ptrdiff_t idx = 0; idx < size; ++idx )
    {
        seg_raw[idx] &= traits::mask;
    }
}
