
/*
    Copyright (C) 2018 University of Leeds
    Copyright (C) 2003 - 2011-02-23, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class stir::KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData

 \author Daniel Deidda

*/


#include "stir/recon_buildblock/KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/stream.h"
#include "stir/info.h"

#ifdef STIR_MPI
#include "stir/recon_buildblock/DistributedCachingInformation.h"
#endif
#include "stir/recon_buildblock/distributable.h"

#include "stir/is_null_ptr.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributed_functions.h"
#endif
#include "stir/info.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::vector;
using std::pair;
using std::ends;
using std::max;
using std::min;
#endif

#include "stir/IndexRange3D.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"

START_NAMESPACE_STIR

template<typename TargetT>
const char * const
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
registered_name =
"KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData";

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_defaults()
{
  base_type::set_defaults();
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::set_defaults ();
  this->num_neighbours=3;

  this->sigma_m=1;
  this->sigma_p=1;
  this->sigma_dp=1;
  this->sigma_dm=1;
  this->anatomical_image_filename="";
  this->only_2D = 0;
  this->kernelised_output_filename_prefix="";


  this->hybrid=0;
  }

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  //PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::initialise_keymap ();

  this->parser.add_start_key("KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters");
  this->parser.add_stop_key("End KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters");
    this->parser.add_key("anatomical image filename",&this->anatomical_image_filename);
  this->parser.add_key("kernelised output filename prefix",&kernelised_output_filename_prefix);
  this->parser.add_key("number of neighbours",&this->num_neighbours);
  this->parser.add_key("number of non-zero feature elements",&this->num_non_zero_feat);
  this->parser.add_key("sigma_m",&this->sigma_m);
  this->parser.add_key("sigma_p",&this->sigma_p);
  this->parser.add_key("sigma_dp",&this->sigma_dp);
  this->parser.add_key("sigma_dm",&this->sigma_dm);
  this->parser.add_key("only_2D",&this->only_2D);
  this->parser.add_key("hybrid",&this->hybrid);
}

template<typename TargetT>
Succeeded
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_up(shared_ptr<TargetT>  const& target_sptr)
{
    base_type::set_up(target_sptr);

   // PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::set_up (target_sptr);
    return Succeeded::yes;
}

template<typename TargetT>
bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
post_processing()
{
    PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::post_processing ();

    this->subiteration_counter=0;
    this->anatomical_sd=0;

  if(!this->only_2D){
   this->num_elem_neighbourhood=this->num_neighbours*this->num_neighbours*this->num_neighbours ;}
  else{
       this->num_elem_neighbourhood=this->num_neighbours*this->num_neighbours ;
  }


  this->anatomical_sptr= (read_from_file<TargetT>(anatomical_image_filename));
  if (this->anatomical_image_filename != "0"){
      set_anatomical_sptr (this->anatomical_sptr);
      info(boost::format("Reading anatomical data '%1%'")
           % anatomical_image_filename  );

      if (is_null_ptr(this->anatomical_sptr))
          {
              error("Failed to read anatomical file 1 %s", anatomical_image_filename.c_str());
              return false;
          }
      estimate_stand_dev_for_anatomical_image(this->anatomical_sd);

      info(boost::format("SD from anatomical image 1 calculated = '%1%'")
           % this->anatomical_sd);


      if(num_non_zero_feat>1){
      shared_ptr<TargetT> normp_sptr(this->anatomical_sptr->get_empty_copy ());
      shared_ptr<TargetT> normm_sptr(this->anatomical_sptr->get_empty_copy ());

      normp_sptr->resize(IndexRange3D(0,0,0,this->num_voxels-1,0,this->num_elem_neighbourhood-1));
      normm_sptr->resize(IndexRange3D(0,0,0,this->num_voxels-1,0,this->num_elem_neighbourhood-1));
      int dimf_col = this->num_non_zero_feat-1;
      int dimf_row=this->num_voxels;

      calculate_norm_const_matrix(*normm_sptr,
                                  dimf_row,
                                  dimf_col);

      info(boost::format("Kernel from anatomical image 1 calculated "));

      this->set_kpnorm_sptr (normp_sptr);
      this->set_kmnorm_sptr (normm_sptr);
      }
  }
    return false;
}

template <typename TargetT>
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData()
{
  this->set_defaults();
}

/***************************************************************
  get_ functions
***************************************************************/

template <typename TargetT>
const std::string
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_anatomical_filename() const
{ return this->anatomical_image_filename; }

template <typename TargetT>
const int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_num_neighbours() const
{ return this->num_neighbours; }

template <typename TargetT>
const int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_num_non_zero_feat() const
{ return this->num_non_zero_feat; }

template <typename TargetT>
const double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_sigma_m() const
{ return this->sigma_m; }

template <typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_sigma_p()
{ return this->sigma_p; }

template <typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_sigma_dp()
{ return this->sigma_dp; }

template <typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_sigma_dm()
{ return this->sigma_dm; }

template <typename TargetT>
const bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_only_2D() const
{ return this->only_2D; }

template <typename TargetT>
bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_hybrid()
{ return this->hybrid; }

template <typename TargetT >
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_kpnorm_sptr()
{ return this->kpnorm_sptr; }

template <typename TargetT >
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_kmnorm_sptr()
{ return this->kmnorm_sptr; }

template <typename TargetT>
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_anatomical_sptr()
{ return this->anatomical_sptr; }


/***************************************************************
  set_ functions
***************************************************************/

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_kpnorm_sptr (shared_ptr<TargetT > &arg)
{
  this->kpnorm_sptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_kmnorm_sptr (shared_ptr<TargetT> &arg)
{
  this->kmnorm_sptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_anatomical_sptr (shared_ptr<TargetT>& arg)
{
  this->anatomical_sptr = arg;
}


/***************************************************************/
// Here start the definition of few functions that calculate the SD of the anatomical image, a norm matrix and
// finally the Kernelised image

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::calculate_norm_matrix(TargetT &normp,
                                                                                                     const int& dimf_row,
                                                                                                     const int& dimf_col,
                                                                                                     const TargetT& pet,
                                                                                                     Array<3,float> distance)
                                           {




                                               Array<2,float> fp;
                                               int l=0,m=0;

                                               fp = Array<2,float>(IndexRange2D(0,dimf_row,0,dimf_col));

                                               const int min_z = pet.get_min_index();
                                               const int max_z = pet.get_max_index();
                                                   this->dimz=max_z-min_z+1;

                                               for (int z=min_z; z<=max_z; z++)
                                                 {
                                                   const int min_dz = max(distance.get_min_index(), min_z-z);
                                                   const int max_dz = min(distance.get_max_index(), max_z-z);

                                                   const int min_y = pet[z].get_min_index();
                                                   const int max_y = pet[z].get_max_index();
                                                     this->dimy=max_y-min_y+1;
                                                     for (int y=min_y;y<= max_y;y++)
                                                       {
                                                         const int min_dy = max(distance[0].get_min_index(), min_y-y);
                                                         const int max_dy = min(distance[0].get_max_index(), max_y-y);

                                                         const int min_x = pet[z][y].get_min_index();
                                                         const int max_x = pet[z][y].get_max_index();
                                                          this->dimx=max_x-min_x+1;


                                           for (int x=min_x;x<= max_x;x++)
                                                           {
                                                             const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
                                                             const int max_dx = min(distance[0][0].get_max_index(), max_x-x);


                                                             l=(z-min_z)*(max_x-min_x +1)*(max_y-min_y +1) + (y-min_y)*(max_x-min_x +1) + (x-min_x);

                                           //here a matrix with the feature vectors is created
                                                             for (int dz=min_dz;dz<=max_dz;++dz)
                                                               for (int dy=min_dy;dy<=max_dy;++dy)
                                                                 for (int dx=min_dx;dx<=max_dx;++dx)
                                                                   {
                                                                     m=(dz)*(max_dx-min_dx +1)*(max_dy-min_dy +1) + (dy)*(max_dx-min_dx +1) + (dx);
                                                                     int c=m;
                                                                     if(m<0){
                                                                         c=m+this->num_elem_neighbourhood ;
                                                                     }else{c=m;}

                                                                     if (z+dz > max_z || y+dy> max_y || x+dx > max_x || z+dz < min_z || y+dy< min_y || x+dx < min_x || m > this->num_non_zero_feat-1 || m <0){
                                                                         //std::cout <<" oltre ="<<x+dx+1<<", "<<y+dy+1<<", "<<z+dz<<std::endl;
                                                                         //std::cout <<" max ="<<max_x<<", "<<max_y<<", "<<max_z<<std::endl;
                                                                         //std::cout <<" min ="<<min_x<<", "<<min_y<<", "<<min_z<<std::endl;
                                                                         continue;
                                                                     }
                                                                     else{
                                                                         fp[l][c]= (pet[z+dz][y+dy][x+dx]) ;
                                                                                }
                                                                        }
                                                                    }
                                                             }
                                                       }

                                           //the norms of the difference between feature vectors related to the same neighbourhood are calculated now
                                           int p=0,o=0;

                                                  for (int q=0; q<=dimf_row-1; ++q){
                                                   for (int n=-(this->num_neighbours-1)/2*(!this->only_2D); n<=(this->num_neighbours-1)/2*(!this->only_2D); ++n)
                                                    for (int k=-(this->num_neighbours-1)/2; k<=(this->num_neighbours-1)/2; ++k)
                                                     for (int j=-(this->num_neighbours-1)/2; j<=(this->num_neighbours-1)/2; ++j)
                                                      for (int i=0; i<=dimf_col; ++i)
                                                       {

                                                       p=j+k*(this->num_neighbours)+n*(this->num_neighbours)*(this->num_neighbours)+(this->num_elem_neighbourhood-1)/2;

                                                       if(q%dimx==0 && (j+k*this->dimx+n*dimx*dimy)>=(dimx-1))
                                                          {if(j+k*this->dimx+n*dimx*dimy>=dimx+(this->num_neighbours-1)/2){
                                                               continue;}

                                                             o=q+j+k*this->dimx+n*dimx*dimy+1;}
                                                       else{o=q+j+k*this->dimx+n*dimx*dimy;}

                                                       if(o>=dimf_row-1 || o<0 || i<0|| i>this->num_non_zero_feat-1 || q>=dimf_row-1 || q<0){
                                                           //std::cout <<"i j k ="<<i<<", "<<j<<", "<<k<<std::endl;
                                                           continue;
                                                       }
                                                                  normp[0][q][p]+=square(fp[q][i]-fp[o][i]);
                                                       }
                                                  }

}

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::calculate_norm_const_matrix(TargetT &normm,
                                                          const int& dimf_row,
                                                          const int &dimf_col)
{




    Array<2,float> fm;
    int l=0,m=0;

    fm = Array<2,float>(IndexRange2D(0,dimf_row,0,dimf_col));
    const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical_cast =
       dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->anatomical_sptr.get ());
const CartesianCoordinate3D<float>& grid_spacing = current_anatomical_cast->get_grid_spacing();
int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

     if (only_2D)
       {
         min_dz = max_dz = 0;
       }
     else
       {
         min_dz = -(num_neighbours-1)/2;
         max_dz = (num_neighbours-1)/2;
       }
     min_dy = -(num_neighbours-1)/2;
     max_dy = (num_neighbours-1)/2;
     min_dx = -(num_neighbours-1)/2;
     max_dx = (num_neighbours-1)/2;

Array<3,float> distance = Array<3,float>(IndexRange3D(min_dz,max_dz,min_dy,max_dy,min_dx,max_dx));

     for (int z=min_dz;z<=max_dz;++z)
       for (int y=min_dy;y<=max_dy;++y)
         for (int x=min_dx;x<=max_dx;++x)
           { // the distance is the euclidean distance:
             //at the moment is used only for the definition of the neighbourhood

                 distance[z][y][x] =

                   sqrt(square(x*grid_spacing.x())+
                        square(y*grid_spacing.y())+
                        square(z*grid_spacing.z()));
           }

    const int min_z = (*anatomical_sptr).get_min_index();
    const int max_z = (*anatomical_sptr).get_max_index();
        this->dimz=max_z-min_z+1;

    for (int z=min_z; z<=max_z; z++)
      {
        const int min_dz = max(distance.get_min_index(), min_z-z);
        const int max_dz = min(distance.get_max_index(), max_z-z);

        const int min_y = (*anatomical_sptr)[z].get_min_index();
        const int max_y = (*anatomical_sptr)[z].get_max_index();
          this->dimy=max_y-min_y+1;
          for (int y=min_y;y<= max_y;y++)
            {
              const int min_dy = max(distance[0].get_min_index(), min_y-y);
              const int max_dy = min(distance[0].get_max_index(), max_y-y);

              const int min_x = (*anatomical_sptr)[z][y].get_min_index();
              const int max_x = (*anatomical_sptr)[z][y].get_max_index();
               this->dimx=max_x-min_x+1;


for (int x=min_x;x<= max_x;x++)
                {
                  const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
                  const int max_dx = min(distance[0][0].get_max_index(), max_x-x);

                  l=(z-min_z)*(max_x-min_x +1)*(max_y-min_y +1) + (y-min_y)*(max_x-min_x +1) + (x-min_x);
//                  std::cout <<" l ="<<l<<" minz ="<<min_z<<" miny ="<<max_y<<" minx ="<<max_x<<std::endl;

//here a matrix with the feature vector is created
                  for (int dz=min_dz;dz<=max_dz;++dz)
                    for (int dy=min_dy;dy<=max_dy;++dy)
                      for (int dx=min_dx;dx<=max_dx;++dx)
                        {
                          m=(dz)*(max_dx-min_dx +1)*(max_dy-min_dy +1) + (dy)*(max_dx-min_dx +1) + (dx);
                          int c=m;
                          if(m<0){
                              c=m+this->num_elem_neighbourhood ;
                          }else{c=m;}

                          if (z+dz > max_z || y+dy> max_y || x+dx > max_x || z+dz < min_z || y+dy< min_y || x+dx < min_x || m > this->num_non_zero_feat-1 || m <0){
                              //std::cout <<" oltre ="<<x+dx+1<<", "<<y+dy+1<<", "<<z+dz<<std::endl;
                              //std::cout <<" max ="<<max_x<<", "<<max_y<<", "<<max_z<<std::endl;
                              //std::cout <<" min ="<<min_x<<", "<<min_y<<", "<<min_z<<std::endl;
                              continue;
                          }
                          else{
                                 fm[l][c]= ((*anatomical_sptr)[z+dz][y+dy][x+dx]);
                                }
                             }

                         }
                     }
                }


//the norms of the difference between feature vectors related to the same neighbourhood are calculated now
int p=0,o=0;

    for (int q=0; q<=dimf_row-1; ++q){
     for (int n=-(this->num_neighbours-1)/2*(!this->only_2D); n<=(this->num_neighbours-1)/2*(!this->only_2D); ++n)
      for (int k=-(this->num_neighbours-1)/2; k<=(this->num_neighbours-1)/2; ++k)
       for (int j=-(this->num_neighbours-1)/2; j<=(this->num_neighbours-1)/2; ++j)
        for (int i=0; i<=dimf_col; ++i)
            {

               p=j+k*(this->num_neighbours)+n*(this->num_neighbours)*(this->num_neighbours)+(this->num_elem_neighbourhood-1)/2;

            if(q%dimx==0 && (j+k*this->dimx+n*dimx*dimy)>=(dimx-1))
               {if(j+k*this->dimx+n*dimx*dimy>=dimx+(this->num_neighbours-1)/2){
                    continue;}

                  o=q+j+k*this->dimx+n*dimx*dimy+1;}
            else{o=q+j+k*this->dimx+n*dimx*dimy;}

            if(o>=dimf_row-1 ||o<0 || i<0|| i>this->num_non_zero_feat-1 || q>=dimf_row-1 || q<0){
                //std::cout <<"i j k ="<<i<<", "<<j<<", "<<k<<std::endl;
                continue;
            }

                 normm[0][q][p]+=square(fm[q][i]-fm[o][i]);}
}

}

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::estimate_stand_dev_for_anatomical_image(double& SD)
{
    double kmean=0;
    double kStand_dev=0;
    double dim_z=0;
    int nv=0;
    const int min_z = (*anatomical_sptr).get_min_index();
    const int max_z = (*anatomical_sptr).get_max_index();

     dim_z = max_z -min_z+1;

        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = (*anatomical_sptr)[z].get_min_index();
            const int max_y = (*anatomical_sptr)[z].get_max_index();
            double dim_y=0;

            dim_y = max_y -min_y+1;

              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = (*anatomical_sptr)[z][y].get_min_index();
                  const int max_x = (*anatomical_sptr)[z][y].get_max_index();
                  double dim_x=0;

                  dim_x = max_x -min_x +1;

                   this->num_voxels = dim_z*dim_y*dim_x;

                    for (int x=min_x;x<= max_x;x++)
                    {
                        if((*anatomical_sptr)[z][y][x]>=0 && (*anatomical_sptr)[z][y][x]<=1000000){
                        kmean += (*anatomical_sptr)[z][y][x];
                        nv+=1;}
                        else{
                            error("The anatomical image might contain nan, negatives or infinitive");
                            break;}
                    }
                }
            }
                      kmean=kmean / nv;

                      for (int z=min_z; z<=max_z; z++)
                        {


                          const int min_y = (*anatomical_sptr)[z].get_min_index();
                          const int max_y = (*anatomical_sptr)[z].get_max_index();

                            for (int y=min_y;y<= max_y;y++)
                              {

                                const int min_x = (*anatomical_sptr)[z][y].get_min_index();
                                const int max_x = (*anatomical_sptr)[z][y].get_max_index();

                                for (int x=min_x;x<= max_x;x++)
                                  {
                                    if((*anatomical_sptr)[z][y][x]>=0 && (*anatomical_sptr)[z][y][x]<=1000000){
                                        kStand_dev += square((*anatomical_sptr)[z][y][x] - kmean);}
                                    else{continue;}
                                  }
                               }
                       }

       SD= sqrt(kStand_dev / (nv-1));
}



template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::full_compute_kernelised_image (
                         TargetT& kImage,
                         const TargetT& Image,
                         const TargetT& current_estimate)
{


//  Something very weird happens here if I do not get_empty_copy() KImage elements will be all nan
    unique_ptr<TargetT> kImage_uptr(current_estimate.get_empty_copy());
    kImage=*kImage_uptr;

    const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical_cast =
         dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->get_anatomical_sptr ().get());
    const CartesianCoordinate3D<float>& grid_spacing = current_anatomical_cast->get_grid_spacing();

  double kPET=0;
  int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

  //Daniel: compute distance for voxel in the neighbourhood from anatomical
       if (only_2D)
         {
           min_dz = max_dz = 0;
         }
       else
         {
           min_dz = -(num_neighbours-1)/2;
           max_dz = (num_neighbours-1)/2;
         }
       min_dy = -(num_neighbours-1)/2;
       max_dy = (num_neighbours-1)/2;
       min_dx = -(num_neighbours-1)/2;
       max_dx = (num_neighbours-1)/2;

  Array<3,float> distance = Array<3,float>(IndexRange3D(min_dz,max_dz,min_dy,max_dy,min_dx,max_dx));

       for (int z=min_dz;z<=max_dz;++z)
         for (int y=min_dy;y<=max_dy;++y)
           for (int x=min_dx;x<=max_dx;++x)
             { // the distance is the euclidean distance:
               //at the moment is used only for the definition of the neighbourhood

                   distance[z][y][x] =

                     sqrt(square(x*grid_spacing.x())+
                          square(y*grid_spacing.y())+
                          square(z*grid_spacing.z()));
             }


      int l=0,m=0, dimf_row=0;
      int dimf_col = this->num_non_zero_feat-1;

      dimf_row=this->num_voxels;

       if(this->get_hybrid ()){
       calculate_norm_matrix (*this->kpnorm_sptr,
                              dimf_row,
                              dimf_col,
                               current_estimate,
                               distance);
}




//     calculate kernelised image

       const int min_z = current_estimate.get_min_index();
       const int max_z = current_estimate.get_max_index();

       for (int z=min_z; z<=max_z; z++)
       { double pnkernel=0, kanatomical=0;
         const int min_dz = max(distance.get_min_index(), min_z-z);
         const int max_dz = min(distance.get_max_index(), max_z-z);

         const int min_y = current_estimate[z].get_min_index();
         const int max_y = current_estimate[z].get_max_index();


           for (int y=min_y;y<= max_y;y++)
             {
               const int min_dy = max(distance[0].get_min_index(), min_y-y);
               const int max_dy = min(distance[0].get_max_index(), max_y-y);

               const int min_x = current_estimate[z][y].get_min_index();
               const int max_x = current_estimate[z][y].get_max_index();


for (int x=min_x;x<= max_x;x++)
                 {
                   const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
                   const int max_dx = min(distance[0][0].get_max_index(), max_x-x);


                   l=(z-min_z)*(max_x-min_x +1)*(max_y-min_y +1) + (y-min_y)*(max_x-min_x +1) + (x-min_x);


                   for (int dz=min_dz;dz<=max_dz;++dz)
                     for (int dy=min_dy;dy<=max_dy;++dy)
                       for (int dx=min_dx;dx<=max_dx;++dx)
                         {
                           m=(dz-min_dz)*(max_dx-min_dx +1)*(max_dy-min_dy +1) + (dy-min_dy)*(max_dx-min_dx +1) + (dx-min_dx);

                           if (get_hybrid()){

                               if(current_estimate[z][y][x]==0){
                                    continue;

                               }
                               else{

                               kPET=exp(-(*this->kpnorm_sptr)[0][l][m]/square(current_estimate[z][y][x]*get_sigma_p())/2)*
                                    exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(get_sigma_dp())));
}

                   }
                   else{
                       kPET=1;

                   }

                   kanatomical=exp(-(*this->kmnorm_sptr)[0][l][m]/square(anatomical_sd*sigma_m)/2)*
                                exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(sigma_dm)));

                   kImage[z][y][x] += kanatomical*kPET*Image[z+dz][y+dy][x+dx];

                   pnkernel += kanatomical*kPET;
                  }
                   if(current_estimate[z][y][x]==0){
                        continue;}

                   kImage[z][y][x]=kImage[z][y][x]/pnkernel;
                   pnkernel=0;


              }
           }
     }
}


template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::compact_compute_kernelised_image(
                         TargetT& kImage,
                         const TargetT& Image,
                         const TargetT& current_estimate)
{


//   Something very weird happens here if I do not get_empty_copy() KImage elements will be all nan
    unique_ptr<TargetT> kImage_uptr(current_estimate.get_empty_copy());
    kImage=*kImage_uptr;

      const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical_cast =
         dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->get_anatomical_sptr ().get());
      const CartesianCoordinate3D<float>& grid_spacing = current_anatomical_cast->get_grid_spacing();

      double kPET =0;
      int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

       if (only_2D)
         {
           min_dz = max_dz = 0;
         }
       else
         {
           min_dz = -(num_neighbours-1)/2;
           max_dz = (num_neighbours-1)/2;
         }
       min_dy = -(num_neighbours-1)/2;
       max_dy = (num_neighbours-1)/2;
       min_dx = -(num_neighbours-1)/2;
       max_dx = (num_neighbours-1)/2;

  Array<3,float> distance = Array<3,float>(IndexRange3D(min_dz,max_dz,min_dy,max_dy,min_dx,max_dx));

       for (int z=min_dz;z<=max_dz;++z)
         for (int y=min_dy;y<=max_dy;++y)
           for (int x=min_dx;x<=max_dx;++x)
             { // the distance is the euclidean distance:
               //at the moment is used only for the definition of the neighbourhood

                   distance[z][y][x] =

                     sqrt(square(x*grid_spacing.x())+
                          square(y*grid_spacing.y())+
                          square(z*grid_spacing.z()));
             }

// get anatomical standard deviation over all voxels

// calculate kernelised image

              const int min_z = (*anatomical_sptr).get_min_index();
              const int max_z = (*anatomical_sptr).get_max_index();

              for (int z=min_z; z<=max_z; z++)
                { double pnkernel=0, kanatomical=0;
                  const int min_dz = max(distance.get_min_index(), min_z-z);
                  const int max_dz = min(distance.get_max_index(), max_z-z);

                  const int min_y = (*anatomical_sptr)[z].get_min_index();
                  const int max_y = (*anatomical_sptr)[z].get_max_index();

                    for (int y=min_y;y<= max_y;y++)
                      {
                        const int min_dy = max(distance[0].get_min_index(), min_y-y);
                        const int max_dy = min(distance[0].get_max_index(), max_y-y);

                        const int min_x = (*anatomical_sptr)[z][y].get_min_index();
                        const int max_x = (*anatomical_sptr)[z][y].get_max_index();


       for (int x=min_x;x<= max_x;x++)
                          {
                            const int min_dx = max(distance[0][0].get_min_index(), min_x-x);
                            const int max_dx = min(distance[0][0].get_max_index(), max_x-x);

                            for (int dz=min_dz;dz<=max_dz;++dz)
                              for (int dy=min_dy;dy<=max_dy;++dy)
                                for (int dx=min_dx;dx<=max_dx;++dx)
                                  {
                                    if (get_hybrid()){

                                        if(current_estimate[z][y][x]==0){
                                             continue;

                                        }
                                        else{

                                        kPET=exp(-square((current_estimate[z][y][x]-current_estimate[z+dz][y+dy][x+dx])/current_estimate[z][y][x]*get_sigma_p())/2)*
                                             exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(get_sigma_dp())));
                                        }

                            }
                            else{
                                kPET=1;

                            }  // the following "pnkernel" is the normalisation of the kernel
                                    kanatomical=exp(-square(((*anatomical_sptr)[z][y][x]-(*anatomical_sptr)[z+dz][y+dy][x+dx])/anatomical_sd*sigma_m)/2)*
                                                 exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(sigma_dm)));

                                    pnkernel+=kPET*kanatomical;

                                    kImage[z][y][x] += kanatomical*kPET*Image[z+dz][y+dy][x+dx];//

                                   }
                                    if(current_estimate[z][y][x]==0){
                                         continue;}
                                     kImage[z][y][x]= kImage[z][y][x]/pnkernel;
                                     pnkernel=0;
                       }
                    }
              }

}

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::compute_kernelised_image(
                         TargetT& kImage,
                         const TargetT& Image,
                         const TargetT& current_estimate)
{

    if(this->num_non_zero_feat==1){
        compact_compute_kernelised_image (kImage, Image, current_estimate);
                                    }
    else{
      full_compute_kernelised_image (kImage, Image,
                               current_estimate);
}
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,
                                                      const TargetT &current_estimate,
                                                      const int subset_num)
{


    subiteration_counter+=1;

    unique_ptr<TargetT> kImage_uptr(current_estimate.get_empty_copy());
    TargetT& kImage=*kImage_uptr;


      compute_kernelised_image(kImage, *current_estimate.clone(),
                               current_estimate);

if((subiteration_counter-1)%this->get_num_subsets()==0){

    char itC[10];
    sprintf (itC, "%d", subiteration_counter-1);
    std::string it=itC;
    std::string us="_";
    std::string k="_k.hv";
    this->current_kimage_filename =this->kernelised_output_filename_prefix+us+it+k;

    write_to_file(this->current_kimage_filename,kImage);

}


    const std::string current_sensitivity_filename =
      boost::str(boost::format(this->get_subsensitivity_filenames ()) % subset_num);


         shared_ptr<TargetT> sens_sptr(read_from_file<TargetT>(current_sensitivity_filename));

         shared_ptr<TargetT> ksens_sptr(current_estimate.get_empty_copy());
         TargetT& ksens= *ksens_sptr;

          compute_kernelised_image(ksens, *sens_sptr,
                                          current_estimate);
        *ksens_sptr=ksens;

        this->set_subset_sensitivity_sptr (ksens_sptr,subset_num);

 PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::compute_sub_gradient_without_penalty_plus_sensitivity (
             gradient,
             kImage,
             subset_num);

        unique_ptr<TargetT> gradient_uptr(gradient.clone());
        compute_kernelised_image(gradient,*gradient_uptr,
                                                         current_estimate);
  }


template<typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                  const int subset_num)
{
  double accum=0.;

  unique_ptr<TargetT> kImage_uptr(current_estimate.get_empty_copy());
  TargetT& kImage=*kImage_uptr;

     compute_kernelised_image(kImage, current_estimate,
                             current_estimate);

PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::actual_compute_objective_function_without_penalty (
            kImage,
            subset_num);

  return accum;
}


template class KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR
