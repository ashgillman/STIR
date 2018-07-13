
/*
    Copyright (C) 2018 University of Leeds
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
//#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/stream.h"
#include "stir/info.h"

#include "stir/recon_buildblock/ProjectorByBinPair.h"

#include "stir/DiscretisedDensity.h"
#ifdef STIR_MPI
#include "stir/recon_buildblock/DistributedCachingInformation.h"
#endif
#include "stir/recon_buildblock/distributable.h"
// for get_symmetries_ptr()
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
// include the following to set defaults
#ifndef USE_PMRT
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#else
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"


#include "stir/Viewgram.h"
#include "stir/recon_array_functions.h"
#include "stir/is_null_ptr.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributed_functions.h"
#endif
#include "stir/CPUTimer.h"
#include "stir/info.h"
#include <boost/format.hpp>
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

const int rim_truncation_sino = 0; // TODO get rid of this

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
  this->sens_filenames =this->get_subsensitivity_filenames();
  this->input_filename="";
  this->max_segment_num_to_process=-1;
  // KT 20/06/2001 disabled
  //num_views_to_add=1;
  this->proj_data_sptr.reset(); //MJ added
  this->zero_seg0_end_planes = 0;
  this->neighbours_num=3;

  this->sigma_m=1;
  this->sigma_p=1;
  this->sigma_dp=1;
  this->sigma_dm=1;
  this->anatomical1_image_filename="";
  this->only_2D = 0;
  this->kernelised_output_filename_prefix="";

  this->subiter_num=0;
  this->kSt_dev=0;
  this->hybrid=0;
  this->additive_projection_data_filename = "0";
  this->additive_proj_data_sptr.reset();


  // set default for projector_pair_ptr
#ifndef USE_PMRT
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingRayTracing());
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingInterpolation());
#else
  shared_ptr<ProjMatrixByBinUsingRayTracing> PM(new  ProjMatrixByBinUsingRayTracing());
  // PM->set_num_tangential_LORs(5);
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingProjMatrixByBin(PM));
#endif

  this->projector_pair_ptr.reset(
                 new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));

  this->normalisation_sptr.reset(new TrivialBinNormalisation);
  this->frame_num = 1;
  this->frame_definition_filename = "";
  // make a single frame starting from 0 to 1.
  vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
  this->frame_defs = TimeFrameDefinitions(frame_times);


  // image stuff
  this->output_image_size_xy=-1;
  this->output_image_size_z=-1;
  this->zoom=1.F;
  this->Xoffset=0.F;
  this->Yoffset=0.F;
  // KT 20/06/2001 new
  this->Zoffset=0.F;

#ifdef STIR_MPI
  //distributed stuff
  this->distributed_cache_enabled = false;
  this->distributed_tests_enabled = false;
  this->message_timings_enabled = false;
  this->message_timings_threshold = 0.1;
  this->rpc_timings_enabled = false;
#endif
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters");
  this->parser.add_stop_key("End KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters");
  this->parser.add_key("input file",&this->input_filename);
  // KT 20/06/2001 disabled
  //parser.add_key("mash x views", &num_views_to_add);

  this->parser.add_key("maximum absolute segment number to process", &this->max_segment_num_to_process);
  this->parser.add_key("zero end planes of segment 0", &this->zero_seg0_end_planes);

  // kernel stuff

  this->parser.add_key("anatomical1_image_filename",&this->anatomical1_image_filename);
  this->parser.add_key("kernelised output filename prefix",&kernelised_output_filename_prefix);
  this->parser.add_key("neighbours_num",&this->neighbours_num);
  this->parser.add_key("num_non_zero_feat_elements",&this->num_non_zero_feat);
  this->parser.add_key("sigma_m",&this->sigma_m);
  this->parser.add_key("sigma_p",&this->sigma_p);
  this->parser.add_key("sigma_dp",&this->sigma_dp);
  this->parser.add_key("sigma_dm",&this->sigma_dm);
  this->parser.add_key("only_2D",&this->only_2D);
  this->parser.add_key("hybrid",&this->hybrid);

  // image stuff

  this->parser.add_key("zoom", &this->zoom);
  this->parser.add_key("XY output image size (in pixels)",&this->output_image_size_xy);
  this->parser.add_key("Z output image size (in pixels)",&this->output_image_size_z);
  //parser.add_key("X offset (in mm)", &this->Xoffset); // KT 10122001 added spaces
  //parser.add_key("Y offset (in mm)", &this->Yoffset);

  this->parser.add_key("Z offset (in mm)", &this->Zoffset);

  this->parser.add_parsing_key("Projector pair type", &this->projector_pair_ptr);
  this->parser.add_key("additive sinogram",&this->additive_projection_data_filename);
  // normalisation (and attenuation correction)
  this->parser.add_key("time frame definition filename", &this->frame_definition_filename);
  this->parser.add_key("time frame number", &this->frame_num);
  this->parser.add_parsing_key("Bin Normalisation type", &this->normalisation_sptr);

#ifdef STIR_MPI
  //distributed stuff
  this->parser.add_key("enable distributed caching", &distributed_cache_enabled);
  this->parser.add_key("enable distributed tests", &distributed_tests_enabled);
  this->parser.add_key("enable message timings", &message_timings_enabled);
  this->parser.add_key("message timings threshold", &message_timings_threshold);
  this->parser.add_key("enable rpc timings", &rpc_timings_enabled);
#endif
}

template<typename TargetT>
bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

    // NE 7/7/2017 The input data can be set alternatively from the set_input_data.
  if (this->input_filename.length() == 0)
  {
      warning("You have not specified an input file, \n"
              "please set one later.");
//      return true;
  }
  // KT 20/06/2001 disabled as not functional yet
#if 0
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)"); return true; }
#endif

  if (this->input_filename.length() > 0 )
  {
    this->proj_data_sptr= ProjData::read_from_file(input_filename);

    if (is_null_ptr(this->proj_data_sptr))
        {
            warning("Failed to read input file %s", input_filename.c_str());
            return true;
        }
  }

 // image stuff
  if (this->zoom <= 0)
  { warning("zoom should be positive"); return true; }

  if (this->output_image_size_xy!=-1 && this->output_image_size_xy<1) // KT 10122001 appended_xy
  { warning("output image size xy must be positive (or -1 as default)"); return true; }
  if (this->output_image_size_z!=-1 && this->output_image_size_z<1) // KT 10122001 new
  { warning("output image size z must be positive (or -1 as default)"); return true; }

  if(!this->only_2D){
   this->num_elem_neighbourhood=this->neighbours_num*this->neighbours_num*this->neighbours_num ;}
  else{
       this->num_elem_neighbourhood=this->neighbours_num*this->neighbours_num ;
  }


  this->anatomical1_sptr= (read_from_file<TargetT>(anatomical1_image_filename));
  if (this->anatomical1_image_filename != "0"){
      set_anatomical1_sptr (this->anatomical1_sptr);
      info(boost::format("Reading anatomical1 data '%1%'")
           % anatomical1_image_filename  );
      double SD=0;

      if (is_null_ptr(this->anatomical1_sptr))
          {
              warning("Failed to read anatomical file 1 %s", anatomical1_image_filename.c_str());
              return true;
          }
      estimate_stand_dev_for_anatomical_image(SD);
      this->set_kSD (SD);
      info(boost::format("SD from anatomical image 1 calculated = '%1%'")
           % this->get_kSD ());


      if(num_non_zero_feat>1){
      shared_ptr<TargetT> normp_sptr(this->anatomical1_sptr->get_empty_copy ());
//       TargetT& normp = *this->get_anatomical1_sptr ().get()->get_empty_copy();
       shared_ptr<TargetT> normm_sptr(this->anatomical1_sptr->get_empty_copy ());

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
  if (this->additive_projection_data_filename != "0")
  {
    info(boost::format("Reading additive projdata data %1%") % this->additive_projection_data_filename);
    this->additive_proj_data_sptr =
      ProjData::read_from_file(this->additive_projection_data_filename);
  };

  // read time frame def
   if (this->frame_definition_filename.size()!=0)
    this->frame_defs = TimeFrameDefinitions(this->frame_definition_filename);
   else
    {
      // make a single frame starting from 0 to 1.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
      this->frame_defs = TimeFrameDefinitions(frame_times);
    }
#ifndef STIR_MPI
#if 0
   //check caching enabled value
   if (this->distributed_cache_enabled==true)
     {
       warning("STIR must be compiled with MPI-compiler to use distributed caching.\n\tDistributed Caching support will be disabled!");
       this->distributed_cache_enabled=false;
     }
   //check tests enabled value
   if (this->distributed_tests_enabled==true || rpc_timings_enabled==true || message_timings_enabled==true)
     {
       warning("STIR must be compiled with MPI-compiler and debug symbols to use distributed testing.\n\tDistributed tests will not be performed!");
       this->distributed_tests_enabled=false;
     }
#endif
#else
   //check caching enabled value
   if (this->distributed_cache_enabled==true)
     info("Will use distributed caching!");
   else info("Distributed caching is disabled. Will use standard distributed version without forced caching!");

#ifndef NDEBUG
   //check tests enabled value
   if (this->distributed_tests_enabled==true)
     {
       warning("\nWill perform distributed tests! Beware that this decreases the performance");
       distributed::test=true;
     }
#else
   //check tests enabled value
   if (this->distributed_tests_enabled==true)
     {
       warning("\nDistributed tests only abvailable in debug mode!");
       distributed::test=false;
     }
#endif

   //check timing values
   if (this->message_timings_enabled==true)
     {
       info("Will print timings of MPI-Messages! This is used to find bottlenecks!");
       distributed::test_send_receive_times=true;
     }
   //set timing threshold
   distributed::min_threshold=this->message_timings_threshold;

   if (this->rpc_timings_enabled==true)
     {
       info("Will print run-times of processing RPC_process_related_viewgrams_gradient for every slave! This will give an idea of the parallelization effect!");
       distributed::rpc_time=true;
     }

#endif

   //this->already_setup = false;
   return false;
}

template <typename TargetT>
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData()
{
  this->set_defaults();




}

template <typename TargetT>
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
~KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData()
{
  end_distributable_computation();
}

template <typename TargetT>
TargetT *
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
construct_target_ptr() const
{
  return
      new VoxelsOnCartesianGrid<float> (*this->proj_data_sptr->get_proj_data_info_ptr(),
                                        static_cast<float>(this->zoom),
                                        CartesianCoordinate3D<float>(static_cast<float>(this->Zoffset),
                                                                     static_cast<float>(this->Yoffset),
                                                                     static_cast<float>(this->Xoffset)),
                                        CartesianCoordinate3D<int>(this->output_image_size_z,
                                                                   this->output_image_size_xy,
                                                                   this->output_image_size_xy)
                                       );
}

/***************************************************************
  get_ functions
***************************************************************/

template <typename TargetT>
const std::string
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_anatomical1_filename() const
{ return this->anatomical1_image_filename; }

template <typename TargetT>
const int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_neighbours_num() const
{ return this->neighbours_num; }

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

template <typename TargetT>
int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_subiter_num()
{ return this->subiter_num; }

template <typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_kSD()
{ return this->kSt_dev; }

template <typename TargetT >
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_kpnorm_sptr()
{ return this->kpnorm_sptr; }

template <typename TargetT >
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_kmnorm_sptr()
{ return this->kmnorm_sptr; }

template <typename TargetT>
shared_ptr<TargetT> &KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::get_anatomical1_sptr()
{ return this->anatomical1_sptr; }



template <typename TargetT>
const ProjData&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_proj_data() const
{ return *this->proj_data_sptr; }

template <typename TargetT>
const shared_ptr<ProjData>&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_proj_data_sptr() const
{ return this->proj_data_sptr; }

template <typename TargetT>
const int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_max_segment_num_to_process() const
{ return this->max_segment_num_to_process; }

template <typename TargetT>
const bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_zero_seg0_end_planes() const
{ return this->zero_seg0_end_planes; }

template <typename TargetT>
const ProjData&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_additive_proj_data() const
{ return *this->additive_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<ProjData>&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_additive_proj_data_sptr() const
{ return this->additive_proj_data_sptr; }

template <typename TargetT>
const ProjectorByBinPair&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_projector_pair() const
{ return *this->projector_pair_ptr; }

template <typename TargetT>
const shared_ptr<ProjectorByBinPair>&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_projector_pair_sptr() const
{ return this->projector_pair_ptr; }

template <typename TargetT>
const int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_time_frame_num() const
{ return this->frame_num; }

template <typename TargetT>
const TimeFrameDefinitions&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_time_frame_definitions() const
{ return this->frame_defs; }

template <typename TargetT>
const BinNormalisation&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_normalisation() const
{ return *this->normalisation_sptr; }

template <typename TargetT>
const shared_ptr<BinNormalisation>&
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
get_normalisation_sptr() const
{ return this->normalisation_sptr; }


/***************************************************************
  set_ functions
***************************************************************/

template<typename TargetT>
int
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_num_subsets(const int new_num_subsets)
{
  this->num_subsets = std::max(new_num_subsets,1);
  return this->num_subsets;

}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_subiter_num (int new_subiter_num)
{
  this->subiter_num = new_subiter_num;

}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_kSD (double kSD)
{
  this->kSt_dev = kSD;
}

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
set_anatomical1_sptr (shared_ptr<TargetT>& arg)
{
  this->anatomical1_sptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_proj_data_sptr(const shared_ptr<ProjData>& arg)
{
  this->proj_data_sptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_max_segment_num_to_process(const int arg)
{
  this->max_segment_num_to_process = arg;

}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_zero_seg0_end_planes(const bool arg)
{
  this->zero_seg0_end_planes = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_additive_proj_data_sptr(const shared_ptr<ExamData> &arg)
{
    this->additive_proj_data_sptr = dynamic_pointer_cast<ProjData>(arg);
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_projector_pair_sptr(const shared_ptr<ProjectorByBinPair>& arg)
{
  this->projector_pair_ptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_frame_num(const int arg)
{
  this->frame_num = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_frame_definitions(const TimeFrameDefinitions& arg)
{
  this->frame_defs = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_normalisation_sptr(const shared_ptr<BinNormalisation>& arg)
{
  this->normalisation_sptr = arg;
}

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_input_data(const shared_ptr<ExamData> & arg)
{
    this->proj_data_sptr = dynamic_pointer_cast<ProjData>(arg);
}

/***************************************************************
  subset balancing
 ***************************************************************/

template<typename TargetT>
bool
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{
  assert(this->num_subsets>0);
  const DataSymmetriesForViewSegmentNumbers& symmetries =
    *this->projector_pair_ptr->get_back_projector_sptr()->get_symmetries_used();

  Array<1,int> num_vs_in_subset(this->num_subsets);
  num_vs_in_subset.fill(0);
  for (int subset_num=0; subset_num<this->num_subsets; ++subset_num)
    {
      for (int segment_num = -this->max_segment_num_to_process;
           segment_num <= this->max_segment_num_to_process;
           ++segment_num)
        for (int view_num = this->proj_data_sptr->get_min_view_num() + subset_num;
             view_num <= this->proj_data_sptr->get_max_view_num();
             view_num += this->num_subsets)
          {
            const ViewSegmentNumbers view_segment_num(view_num, segment_num);
            if (!symmetries.is_basic(view_segment_num))
              continue;
            num_vs_in_subset[subset_num] +=
              symmetries.num_related_view_segment_numbers(view_segment_num);
          }
    }
  for (int subset_num=1; subset_num<this->num_subsets; ++subset_num)
    {
      if(num_vs_in_subset[subset_num] != num_vs_in_subset[0])
        {
          std::stringstream str(warning_message);
          str <<"Number of subsets is such that subsets will be very unbalanced.\n"
              << "Number of viewgrams in each subset would be:\n"
              << num_vs_in_subset
              << "\nEither reduce the number of symmetries used by the projector, or\n"
            "change the number of subsets. It usually should be a divisor of\n"
              << this->proj_data_sptr->get_num_views()
              << "/4 (or if that's not an integer, a divisor of "
              << this->proj_data_sptr->get_num_views()
              << "/2 or "
              << this->proj_data_sptr->get_num_views()
          << ").\n";
          warning_message = str.str();
          return false;
        }
    }
  return true;
}

/***************************************************************
  set_up()
***************************************************************/
template<typename TargetT>
Succeeded
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr)
{
  if (this->max_segment_num_to_process==-1)
    this->max_segment_num_to_process =
      this->proj_data_sptr->get_max_segment_num();

  if (this->max_segment_num_to_process > this->proj_data_sptr->get_max_segment_num())
    {
      warning("max_segment_num_to_process (%d) is too large",
              this->max_segment_num_to_process);
      return Succeeded::no;
    }

  shared_ptr<ProjDataInfo> proj_data_info_sptr(this->proj_data_sptr->get_proj_data_info_ptr()->clone());

  proj_data_info_sptr->
    reduce_segment_range(-this->max_segment_num_to_process,
                         +this->max_segment_num_to_process);

  if (is_null_ptr(this->projector_pair_ptr))
    { warning("You need to specify a projector pair"); return Succeeded::no; }

  // set projectors to be used for the calculations

  setup_distributable_computation(this->projector_pair_ptr,
                                  this->proj_data_sptr->get_exam_info_sptr(),
                                  this->proj_data_sptr->get_proj_data_info_ptr(),
                                  target_sptr,
                                  zero_seg0_end_planes,
                                  distributed_cache_enabled);

#ifdef STIR_MPI
  //set up distributed caching object
  if (distributed_cache_enabled)
    {
      this->caching_info_ptr = new DistributedCachingInformation(distributed::num_processors);
    }
  else caching_info_ptr = NULL;
#else
  //non parallel version
  caching_info_ptr = NULL;
#endif

  this->projector_pair_ptr->set_up(proj_data_info_sptr,
                                   target_sptr);

  // TODO check compatibility between symmetries for forward and backprojector
  this->symmetries_sptr.reset(
                  this->projector_pair_ptr->get_back_projector_sptr()->get_symmetries_used()->clone());

  if (is_null_ptr(this->normalisation_sptr))
  {
    warning("Invalid normalisation object");
    return Succeeded::no;
  }

  if (this->normalisation_sptr->set_up(proj_data_info_sptr) == Succeeded::no)
    return Succeeded::no;

  if (frame_num<=0)
    {
      warning("frame_num should be >= 1");
      return Succeeded::no;
    }

  if (static_cast<unsigned>(frame_num)> frame_defs.get_num_frames())
    {
      warning("frame_num is %d, but should be less than the number of frames %d.",
              frame_num, frame_defs.get_num_frames());
      return Succeeded::no;
    }

  return Succeeded::yes;
}






/***************************************************************
  functions that compute the value/gradient of the objective function etc
***************************************************************/
// Here start the definition of few functions that calculate the SD of the anatomical1 image, a norm matrix and
// finally the Kernelised image

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::calculate_norm_matrix(TargetT &normp,
                                                                                                     const int& dimf_row,
                                                                                                     int& dimf_col,
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

                                           //std::cout <<" row ="<<dimf_row<<" col ="<<dimf_col<<std::endl;

                                                             l=(z-min_z)*(max_x-min_x +1)*(max_y-min_y +1) + (y-min_y)*(max_x-min_x +1) + (x-min_x);
                                           //std::cout <<" l ="<<l<<" minz ="<<min_z<<" miny ="<<max_y<<" minx ="<<max_x<<std::endl;
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
                                                                         fp[l][c]= (pet[z+dz][y+dy][x+dx]) ;
                                                                                }
                                                                        }
                                                                    }
                                                             }
                                                       }

                                           //the norms of the difference between feature vectors related to the same neighbourhood are calculated now
                                           int p=0,o=0;

                                                  for (int q=0; q<=dimf_row-1; ++q){
                                                   for (int n=-(this->neighbours_num-1)/2*(!this->only_2D); n<=(this->neighbours_num-1)/2*(!this->only_2D); ++n)
                                                    for (int k=-(this->neighbours_num-1)/2; k<=(this->neighbours_num-1)/2; ++k)
                                                     for (int j=-(this->neighbours_num-1)/2; j<=(this->neighbours_num-1)/2; ++j)
                                                      for (int i=0; i<=dimf_col; ++i)
                                                       {

                                                       p=j+k*(this->neighbours_num)+n*(this->neighbours_num)*(this->neighbours_num)+(this->num_elem_neighbourhood-1)/2;

                                                       if(q%dimx==0 && (j+k*this->dimx+n*dimx*dimy)>=(dimx-1))
                                                          {if(j+k*this->dimx+n*dimx*dimy>=dimx+(this->neighbours_num-1)/2){
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
                                                          int& dimf_col)
{




    Array<2,float> fm;
    int l=0,m=0;

    fm = Array<2,float>(IndexRange2D(0,dimf_row,0,dimf_col));
    const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical1_cast =
       dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->anatomical1_sptr.get ());
const CartesianCoordinate3D<float>& grid_spacing = current_anatomical1_cast->get_grid_spacing();
int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

     if (only_2D)
       {
         min_dz = max_dz = 0;
       }
     else
       {
         min_dz = -(neighbours_num-1)/2;
         max_dz = (neighbours_num-1)/2;
       }
     min_dy = -(neighbours_num-1)/2;
     max_dy = (neighbours_num-1)/2;
     min_dx = -(neighbours_num-1)/2;
     max_dx = (neighbours_num-1)/2;

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

    const int min_z = (*anatomical1_sptr).get_min_index();
    const int max_z = (*anatomical1_sptr).get_max_index();
        this->dimz=max_z-min_z+1;

    for (int z=min_z; z<=max_z; z++)
      {
        const int min_dz = max(distance.get_min_index(), min_z-z);
        const int max_dz = min(distance.get_max_index(), max_z-z);

        const int min_y = (*anatomical1_sptr)[z].get_min_index();
        const int max_y = (*anatomical1_sptr)[z].get_max_index();
          this->dimy=max_y-min_y+1;
          for (int y=min_y;y<= max_y;y++)
            {
              const int min_dy = max(distance[0].get_min_index(), min_y-y);
              const int max_dy = min(distance[0].get_max_index(), max_y-y);

              const int min_x = (*anatomical1_sptr)[z][y].get_min_index();
              const int max_x = (*anatomical1_sptr)[z][y].get_max_index();
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
                                 fm[l][c]= ((*anatomical1_sptr)[z+dz][y+dy][x+dx]);
                                }
                             }

                         }

                 // std::cout <<l<<std::endl;
                     }
                }


//the norms of the difference between feature vectors related to the same neighbourhood are calculated now
int p=0,o=0;

    for (int q=0; q<=dimf_row-1; ++q){
     for (int n=-(this->neighbours_num-1)/2*(!this->only_2D); n<=(this->neighbours_num-1)/2*(!this->only_2D); ++n)
      for (int k=-(this->neighbours_num-1)/2; k<=(this->neighbours_num-1)/2; ++k)
       for (int j=-(this->neighbours_num-1)/2; j<=(this->neighbours_num-1)/2; ++j)
        for (int i=0; i<=dimf_col; ++i)
            {

               p=j+k*(this->neighbours_num)+n*(this->neighbours_num)*(this->neighbours_num)+(this->num_elem_neighbourhood-1)/2;

            if(q%dimx==0 && (j+k*this->dimx+n*dimx*dimy)>=(dimx-1))
               {if(j+k*this->dimx+n*dimx*dimy>=dimx+(this->neighbours_num-1)/2){
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
    const int min_z = (*anatomical1_sptr).get_min_index();
    const int max_z = (*anatomical1_sptr).get_max_index();

     dim_z = max_z -min_z+1;

        for (int z=min_z; z<=max_z; z++)
          {

            const int min_y = (*anatomical1_sptr)[z].get_min_index();
            const int max_y = (*anatomical1_sptr)[z].get_max_index();
            double dim_y=0;

            dim_y = max_y -min_y+1;

              for (int y=min_y;y<= max_y;y++)
                {

                  const int min_x = (*anatomical1_sptr)[z][y].get_min_index();
                  const int max_x = (*anatomical1_sptr)[z][y].get_max_index();
                  double dim_x=0;

                  dim_x = max_x -min_x +1;

                   this->num_voxels = dim_z*dim_y*dim_x;

                    for (int x=min_x;x<= max_x;x++)
                    {
                        if((*anatomical1_sptr)[z][y][x]>=0 && (*anatomical1_sptr)[z][y][x]<=1000000){
    //                        std::cout <<"i j k ="<<(*anatomical1_sptr)[z][y][x]<<std::endl;
                        kmean += (*anatomical1_sptr)[z][y][x];
                        nv+=1;}
                        else{
                            continue;}
                    }
                }
            }
                      kmean=kmean / nv;

                      for (int z=min_z; z<=max_z; z++)
                        {


                          const int min_y = (*anatomical1_sptr)[z].get_min_index();
                          const int max_y = (*anatomical1_sptr)[z].get_max_index();

                            for (int y=min_y;y<= max_y;y++)
                              {

                                const int min_x = (*anatomical1_sptr)[z][y].get_min_index();
                                const int max_x = (*anatomical1_sptr)[z][y].get_max_index();

                                for (int x=min_x;x<= max_x;x++)
                                  {
                                    if((*anatomical1_sptr)[z][y][x]>=0 && (*anatomical1_sptr)[z][y][x]<=1000000){
                                        kStand_dev += square((*anatomical1_sptr)[z][y][x] - kmean);}
                                    else{continue;}
                                  }
                               }
                       }

       SD= sqrt(kStand_dev / (nv-1));
}

template<typename TargetT>
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::compute_kernelised_image(
                         TargetT& kImage,
                         TargetT& Image,
                         double sigma_m,
                         int neighbours_num,
                         const TargetT& current_estimate,
                         bool only_2D)
{




//  Something very weird happens here if I do not get_empty_copy() KImage elements will be all nan
    kImage= *current_estimate.get_empty_copy();

    const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical1_cast =
         dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->get_anatomical1_sptr ().get());
    const CartesianCoordinate3D<float>& grid_spacing = current_anatomical1_cast->get_grid_spacing();

  double kPET=0;
  int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

  //Daniel: compute distance for voxel in the neighbourhood from anatomical1
       if (only_2D)
         {
           min_dz = max_dz = 0;
         }
       else
         {
           min_dz = -(neighbours_num-1)/2;
           max_dz = (neighbours_num-1)/2;
         }
       min_dy = -(neighbours_num-1)/2;
       max_dy = (neighbours_num-1)/2;
       min_dx = -(neighbours_num-1)/2;
       max_dx = (neighbours_num-1)/2;

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
      double kSD=0, pnkernel=0;

      kSD=this->get_kSD ();

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
       { double pnkernel=0, kAnatomical1=0;
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

                   kAnatomical1=exp(-(*this->kmnorm_sptr)[0][l][m]/square(kSD*sigma_m)/2)*
                                exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(sigma_dm)));

                   kImage[z][y][x] += kAnatomical1*kPET*Image[z+dz][y+dy][x+dx];

                   pnkernel += kAnatomical1*kPET;
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
void KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::fast_compute_kernelised_image(
                         TargetT& kImage,
                         TargetT& Image,
                         double sigma_m,
                         int neighbours_num,
                         const TargetT& current_estimate,
                         bool only_2D)
{


//   Something very weird happens here if I do not get_empty_copy() KImage elements will be all nan
     kImage=*current_estimate.get_empty_copy();
//   write_to_file("kImage1", kImage);

      const DiscretisedDensityOnCartesianGrid<3,float>* current_anatomical1_cast =
         dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> *>(this->get_anatomical1_sptr ().get());
      const CartesianCoordinate3D<float>& grid_spacing = current_anatomical1_cast->get_grid_spacing();

      double kPET =0;
      int min_dz, max_dz,min_dx,max_dx, min_dy,max_dy;

       if (only_2D)
         {
           min_dz = max_dz = 0;
         }
       else
         {
           min_dz = -(neighbours_num-1)/2;
           max_dz = (neighbours_num-1)/2;
         }
       min_dy = -(neighbours_num-1)/2;
       max_dy = (neighbours_num-1)/2;
       min_dx = -(neighbours_num-1)/2;
       max_dx = (neighbours_num-1)/2;

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

// get anatomical1 standard deviation over all voxels
     double kSD=0;
     kSD=get_kSD();

// calculate kernelised image

              const int min_z = (*anatomical1_sptr).get_min_index();
              const int max_z = (*anatomical1_sptr).get_max_index();

              for (int z=min_z; z<=max_z; z++)
                { double pnkernel=0, kAnatomical1=0;
                  const int min_dz = max(distance.get_min_index(), min_z-z);
                  const int max_dz = min(distance.get_max_index(), max_z-z);

                  const int min_y = (*anatomical1_sptr)[z].get_min_index();
                  const int max_y = (*anatomical1_sptr)[z].get_max_index();

                    for (int y=min_y;y<= max_y;y++)
                      {
                        const int min_dy = max(distance[0].get_min_index(), min_y-y);
                        const int max_dy = min(distance[0].get_max_index(), max_y-y);

                        const int min_x = (*anatomical1_sptr)[z][y].get_min_index();
                        const int max_x = (*anatomical1_sptr)[z][y].get_max_index();


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
                                    kAnatomical1=exp(-square(((*anatomical1_sptr)[z][y][x]-(*anatomical1_sptr)[z+dz][y+dy][x+dx])/kSD*sigma_m)/2)*
                                                 exp(-square(distance[dz][dy][dx]/grid_spacing.x ())/(2*square(sigma_dm)));

                                    pnkernel+=kPET*kAnatomical1;

                                    kImage[z][y][x] += kAnatomical1*kPET*Image[z+dz][y+dy][x+dx];//

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
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,
                                                      const TargetT &current_estimate,
                                                      const int subset_num)
{


    set_subiter_num (this->subiter_num+=1);

    TargetT& kImage= *current_estimate.get_empty_copy();

    if(this->num_non_zero_feat==1){
        fast_compute_kernelised_image (kImage, *current_estimate.clone(), sigma_m,
                                                                        neighbours_num,
                                                                        current_estimate,
                                                                        only_2D);
                                    }
    else{
      compute_kernelised_image(kImage, *current_estimate.clone(), sigma_m,
                               neighbours_num,
                               current_estimate,
                               only_2D);
    }
if((get_subiter_num ()-1)%this->get_num_subsets()==0){
//        this->current_kimage_filename =
//          boost::str(boost::format(this->kernelised_output_filename_prefix) %get_subiter_num ());

    char itC[10];
    sprintf (itC, "%d", get_subiter_num ()-1);
    std::string it=itC;
    std::string us="_";
    std::string k="_k.hv";
    this->current_kimage_filename =this->kernelised_output_filename_prefix+us+it+k;
std::cout<<this->current_kimage_filename<<std::endl;
    write_to_file(this->current_kimage_filename,kImage);


//    this->current_kimage_filename =  ;
}


    const std::string current_sensitivity_filename =
      boost::str(boost::format(this->get_subsensitivity_filenames ()) % subset_num);

    //if(get_subiter_num ()<=this->get_num_subsets()){
         shared_ptr<TargetT> sens_sptr(read_from_file<TargetT>(current_sensitivity_filename));

         TargetT& ksens= *current_estimate.get_empty_copy();

         //write_to_file("sens1", ksens);


         if(this->num_non_zero_feat==1){
             fast_compute_kernelised_image(ksens, *sens_sptr,  sigma_m,
                                                              neighbours_num,
                                                              current_estimate,
                                                                    only_2D);
         }
         else{
                 compute_kernelised_image(ksens, *sens_sptr, sigma_m,
                                                                  neighbours_num,
                                                                 current_estimate,
                                                                        only_2D);

         }


        this->set_subset_sensitivity_sptr (shared_ptr< TargetT >(ksens.clone()),subset_num);

  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);
  distributable_compute_gradient1(this->projector_pair_ptr->get_forward_projector_sptr(),
                                 this->projector_pair_ptr->get_back_projector_sptr(),
                                 this->symmetries_sptr,
                                 gradient,
                                 kImage,
                                 this->proj_data_sptr,
                                 subset_num,
                                 this->num_subsets,
                                 -this->max_segment_num_to_process,
                                 this->max_segment_num_to_process,
                                 this->zero_seg0_end_planes!=0,
                                 NULL,
                                 this->additive_proj_data_sptr
                                 , caching_info_ptr
                                 );

  //write_to_file("gradient", gradient);
  if(this->num_non_zero_feat==1){
      fast_compute_kernelised_image(gradient,*gradient.clone(),sigma_m,
                                                                      neighbours_num,
                                                                      current_estimate,
                                                                      only_2D);

  }
  else{
      compute_kernelised_image(gradient,*gradient.clone(),sigma_m,
                                                                  neighbours_num,
                                                                  current_estimate,
                                                                  only_2D);
  }

//write_to_file("kgradient", gradient);
}


template<typename TargetT>
double
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                  const int subset_num)
{
  double accum=0.;

  distributable_accumulate_loglikelihood1(this->projector_pair_ptr->get_forward_projector_sptr(),
                                         this->projector_pair_ptr->get_back_projector_sptr(),
                                         this->symmetries_sptr,
                                         current_estimate,
                                         this->proj_data_sptr,
                                         subset_num, this->get_num_subsets(),
                                         -this->max_segment_num_to_process,
                                         this->max_segment_num_to_process,
                                         this->zero_seg0_end_planes != 0, &accum,
                                         this->additive_proj_data_sptr,
                                         this->normalisation_sptr,
                                         this->get_time_frame_definitions().get_start_time(this->get_time_frame_num()),
                                         this->get_time_frame_definitions().get_end_time(this->get_time_frame_num()),
                                         this->caching_info_ptr
                                         );


  return accum;
}

#if 0
template<typename TargetT>
float
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
sum_projection_data() const
{

  float counts=0.0F;

  for (int segment_num = -max_segment_num_to_process; segment_num <= max_segment_num_to_process; segment_num++)
  {
    for (int view_num = proj_data_sptr->get_min_view_num();
         view_num <= proj_data_sptr->get_max_view_num();
         ++view_num)
    {

      Viewgram<float>  viewgram=proj_data_sptr->get_viewgram(view_num,segment_num);

      //first adjust data

      // KT 05/07/2000 made parameters.zero_seg0_end_planes int
      if(segment_num==0 && zero_seg0_end_planes!=0)
      {
        viewgram[viewgram.get_min_axial_pos_num()].fill(0);
        viewgram[viewgram.get_max_axial_pos_num()].fill(0);
      }

      truncate_rim(viewgram,rim_truncation_sino);

      //now take totals
      counts+=viewgram.sum();
    }
  }

  return counts;

}

#endif

template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  const int min_segment_num = -this->max_segment_num_to_process;
  const int max_segment_num = this->max_segment_num_to_process;

  // warning: has to be same as subset scheme used as in distributable_computation
  for (int segment_num = min_segment_num; segment_num <= max_segment_num; ++segment_num)
  {
        //CPUTimer timer;
        //timer.start();

    for (int view = this->proj_data_sptr->get_min_view_num() + subset_num;
        view <= this->proj_data_sptr->get_max_view_num();
        view += this->num_subsets)
    {
      const ViewSegmentNumbers view_segment_num(view, segment_num);

      if (!symmetries_sptr->is_basic(view_segment_num))
        continue;
      this->add_view_seg_to_sensitivity(sensitivity, view_segment_num);
    }
      //    cerr<<timer.value()<<endl;
  }

}


template<typename TargetT>
void
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
add_view_seg_to_sensitivity(TargetT& sensitivity, const ViewSegmentNumbers& view_seg_nums) const
{
  RelatedViewgrams<float> viewgrams =
    this->proj_data_sptr->get_empty_related_viewgrams(view_seg_nums,
                                                      this->symmetries_sptr);
  viewgrams.fill(1.F);
  // find efficiencies
  {
    const double start_frame = this->frame_defs.get_start_time(this->frame_num);
    const double end_frame = this->frame_defs.get_end_time(this->frame_num);
    this->normalisation_sptr->undo(viewgrams,start_frame,end_frame);
  }
  // backproject
  {
    const int range_to_zero =
      view_seg_nums.segment_num() == 0 && this->zero_seg0_end_planes
      ? 1 : 0;
    const int min_ax_pos_num =
      viewgrams.get_min_axial_pos_num() + range_to_zero;
    const int max_ax_pos_num =
       viewgrams.get_max_axial_pos_num() - range_to_zero;

    this->projector_pair_ptr->get_back_projector_sptr()->
      back_project(sensitivity, viewgrams,
                   min_ax_pos_num, max_ax_pos_num);

  }

}

template<typename TargetT>
Succeeded
KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>::
actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                       const TargetT& input,
                                                                       const int subset_num) const
{
  {
    std::string explanation;
    if (!input.has_same_characteristics(this->get_sensitivity(),
                                        explanation))
      {
        warning("KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData:\n"
                "sensitivity and input for add_multiplication_with_approximate_Hessian_without_penalty\n"
                "should have the same characteristics.\n%s",
                explanation.c_str());
        return Succeeded::no;
      }
  }

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(
    this->get_projector_pair().get_symmetries_used()->clone());

  const double start_time =
    this->get_time_frame_definitions().get_start_time(this->get_time_frame_num());
  const double end_time =
    this->get_time_frame_definitions().get_end_time(this->get_time_frame_num());

  for (int segment_num = -this->get_max_segment_num_to_process();
       segment_num<= this->get_max_segment_num_to_process();
       ++segment_num)
    {
      for (int view = this->get_proj_data().get_min_view_num() + subset_num;
           view <= this->get_proj_data().get_max_view_num();
           view += this->num_subsets)
        {
          const ViewSegmentNumbers view_segment_num(view, segment_num);

          if (!symmetries_sptr->is_basic(view_segment_num))
            continue;

          // first compute data-term: y*norm^2
          RelatedViewgrams<float> viewgrams =
            this->get_proj_data().get_related_viewgrams(view_segment_num, symmetries_sptr);
          // TODO add 1 for 1/(y+1) approximation

          this->get_normalisation().apply(viewgrams, start_time, end_time);

          // smooth TODO

          this->get_normalisation().apply(viewgrams, start_time, end_time);

          RelatedViewgrams<float> tmp_viewgrams;
          // set tmp_viewgrams to geometric forward projection of input
          {
            tmp_viewgrams = this->get_proj_data().get_empty_related_viewgrams(view_segment_num, symmetries_sptr);
            this->get_projector_pair().get_forward_projector_sptr()->
              forward_project(tmp_viewgrams, input);
          }

          // now divide by the data term
          {
            int tmp1=0, tmp2=0;// ignore counters returned by divide_and_truncate
            divide_and_truncate(tmp_viewgrams, viewgrams, 0, tmp1, tmp2);
          }

          // back-project
          this->get_projector_pair().get_back_projector_sptr()->
            back_project(output, tmp_viewgrams);
      }

  } // end of loop over segments

  return Succeeded::yes;
}

/*********************** distributable_* ***************************/
// TODO all this stuff is specific to DiscretisedDensity, so wouldn't work for TargetT

#ifdef STIR_MPI
// make call-backs public for the moment

//! Call-back function for compute_gradient
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_gradient;

//! Call-back function for accumulate_loglikelihood
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_accumulate_loglikelihood;
#else
//! Call-back function for compute_gradient
static RPC_process_related_viewgrams_type RPC_process_related_viewgrams_gradient;

//! Call-back function for accumulate_loglikelihood
static RPC_process_related_viewgrams_type RPC_process_related_viewgrams_accumulate_loglikelihood;
#endif

void distributable_compute_gradient1(const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                    const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                                    const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_sptr,
                                    DiscretisedDensity<3,float>& output_image,
                                    const DiscretisedDensity<3,float>& input_image,
                                    const shared_ptr<ProjData>& proj_dat,
                                    int subset_num, int num_subsets,
                                    int min_segment, int max_segment,
                                    bool zero_seg0_end_planes,
                                    double* log_likelihood_ptr,
                                    shared_ptr<ProjData> const& additive_binwise_correction,
                                    DistributedCachingInformation* caching_info_ptr
                                    )
{

    distributable_computation(forward_projector_sptr,
                              back_projector_sptr,
                              symmetries_sptr,
                              &output_image, &input_image,
                              proj_dat, true, //i.e. do read projection data
                              subset_num, num_subsets,
                              min_segment, max_segment,
                              zero_seg0_end_planes,
                              log_likelihood_ptr,
                              additive_binwise_correction,
                              /* normalisation info to be ignored */ shared_ptr<BinNormalisation>(), 0., 0.,
                              &RPC_process_related_viewgrams_gradient,
                              caching_info_ptr
                              );
}


void distributable_accumulate_loglikelihood1(
                                            const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                            const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                                            const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_sptr,
                                            const DiscretisedDensity<3,float>& input_image,
                                            const shared_ptr<ProjData>& proj_dat,
                                            int subset_num, int num_subsets,
                                            int min_segment, int max_segment,
                                            bool zero_seg0_end_planes,
                                            double* log_likelihood_ptr,
                                            shared_ptr<ProjData> const& additive_binwise_correction,
                                            shared_ptr<BinNormalisation> const& normalisation_sptr,
                                            const double start_time_of_frame,
                                            const double end_time_of_frame,
                                            DistributedCachingInformation* caching_info_ptr
                                            )

{
          distributable_computation(forward_projector_sptr,
                                    back_projector_sptr,
                                    symmetries_sptr,
                                    NULL, &input_image,
                                    proj_dat, true, //i.e. do read projection data
                                    subset_num, num_subsets,
                                    min_segment, max_segment,
                                    zero_seg0_end_planes,
                                    log_likelihood_ptr,
                                    additive_binwise_correction,
                                    normalisation_sptr,
                                    start_time_of_frame,
                                    end_time_of_frame,
                                    &RPC_process_related_viewgrams_accumulate_loglikelihood,
                                    caching_info_ptr
                                    );
}

//////////// RPC functions


void RPC_process_related_viewgrams_gradient(
                                            const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                            const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                                            DiscretisedDensity<3,float>* output_image_ptr,
                                            const DiscretisedDensity<3,float>* input_image_ptr,
                                            RelatedViewgrams<float>* measured_viewgrams_ptr,
                                            int& count, int& count2, double* log_likelihood_ptr /* = NULL */,
                                            const RelatedViewgrams<float>* additive_binwise_correction_ptr,
                                            const RelatedViewgrams<float>* mult_viewgrams_ptr)
{
  assert(output_image_ptr != NULL);
  assert(input_image_ptr != NULL);
  assert(measured_viewgrams_ptr != NULL);
  if (!is_null_ptr(mult_viewgrams_ptr))
    error("Internal error: mult_viewgrams_ptr should be zero when computing gradient");

  RelatedViewgrams<float> estimated_viewgrams = measured_viewgrams_ptr->get_empty_copy();

  /*if (distributed::first_iteration)
    {
        stir::RelatedViewgrams<float>::iterator viewgrams_iter = measured_viewgrams_ptr->begin();
                stir::RelatedViewgrams<float>::iterator viewgrams_end = measured_viewgrams_ptr->end();
                while (viewgrams_iter!= viewgrams_end)
                {
                        printf("\nSLAVE VIEWGRAM\n");
                        int pos=0;
                        for ( int tang_pos = -144 ;tang_pos  <= 143 ;++tang_pos)
                        for ( int ax_pos = 0; ax_pos <= 62 ;++ax_pos)
                        {
                                        if (pos>3616 && pos <3632) printf("%f, ",(*viewgrams_iter)[ax_pos][tang_pos]);
                                        pos++;
                        }
                        viewgrams_iter++;
                }
    }
*/

  forward_projector_sptr->forward_project(estimated_viewgrams, *input_image_ptr);



  if (additive_binwise_correction_ptr != NULL)
  {
    estimated_viewgrams += (*additive_binwise_correction_ptr);
  }






  // for sinogram division

  divide_and_truncate(*measured_viewgrams_ptr, estimated_viewgrams, rim_truncation_sino, count, count2, log_likelihood_ptr);

  back_projector_sptr->back_project(*output_image_ptr, *measured_viewgrams_ptr);
};


void RPC_process_related_viewgrams_accumulate_loglikelihood(
                                                            const shared_ptr<ForwardProjectorByBin>& forward_projector_sptr,
                                                            const shared_ptr<BackProjectorByBin>& back_projector_sptr,
                                                            DiscretisedDensity<3,float>* output_image_ptr,
                                                            const DiscretisedDensity<3,float>* input_image_ptr,
                                                            RelatedViewgrams<float>* measured_viewgrams_ptr,
                                                            int& count, int& count2, double* log_likelihood_ptr,
                                                            const RelatedViewgrams<float>* additive_binwise_correction_ptr,
                                                            const RelatedViewgrams<float>* mult_viewgrams_ptr)
{

  assert(output_image_ptr == NULL);
  assert(input_image_ptr != NULL);
  assert(measured_viewgrams_ptr != NULL);
  assert(log_likelihood_ptr != NULL);

  RelatedViewgrams<float> estimated_viewgrams = measured_viewgrams_ptr->get_empty_copy();

  forward_projector_sptr->forward_project(estimated_viewgrams, *input_image_ptr);

  if (additive_binwise_correction_ptr != NULL)
  {
    estimated_viewgrams += (*additive_binwise_correction_ptr);
  };

  if (mult_viewgrams_ptr != NULL)
  {
    estimated_viewgrams *= (*mult_viewgrams_ptr);
  }

  RelatedViewgrams<float>::iterator meas_viewgrams_iter =
          measured_viewgrams_ptr->begin();
  RelatedViewgrams<float>::const_iterator est_viewgrams_iter =
          estimated_viewgrams.begin();
  // call function that does the actual work, it sits in recon_array_funtions.cxx (TODO)
  for (;
       meas_viewgrams_iter != measured_viewgrams_ptr->end();
       ++meas_viewgrams_iter, ++est_viewgrams_iter)
    accumulate_loglikelihood(*meas_viewgrams_iter,
                             *est_viewgrams_iter,
                             rim_truncation_sino, log_likelihood_ptr);
};


#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable:4661)
#  endif

template class KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR
