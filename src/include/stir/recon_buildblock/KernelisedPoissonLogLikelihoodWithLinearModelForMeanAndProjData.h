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

#ifndef __stir_recon_buildblock_KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__
#define __stir_recon_buildblock_KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
//#include "stir/ProjData.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
//#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/TimeFrameDefinitions.h"
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributable.h" // for  RPC_process_related_viewgrams_type
#endif
#include "stir/Array.h"
//#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>
//#include "stir/recon_buildblock/IterativeReconstruction.h"
START_NAMESPACE_STIR

class DistributedCachingInformation;

//#ifdef STIR_MPI_CLASS_DEFINITION
//#define KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData_MPI
//#endif


/*!
  \ingroup GeneralisedObjectiveFunction
  \brief An objective function class appropriate for PET emission data

  Measured data is given by a ProjData object, and the linear operations
  necessary for computing the gradient of the objective function
  are performed via a ProjectorByBinPair object together with
  a BinNormalisation object.

  \see PoissonLogLikelihoodWithLinearModelForMean.

  This class implements the objective function obtained using the Kernel method (KEM) and Hybrid kernel method (HKEM).
  This implementation corresponds to the one presented by Deidda D et al, ``Hybrid PET-MR list-mode kernelized expectation
  maximization reconstruction for quantitative PET images of the carotid arteries", IEEE MIC Atlanta, 2017. However, this is
  the sinogram-based obgective function. Each voxel value of the image, \f$ \boldsymbol{\lambda}\f$, can be represented as a
  linear combination using the kernel method. % If we have an image with prior information, we can construct for each voxel
  \f$ j \f$ of the PET image a feature vector, $\f \boldsymbol{v}_j \f$, using the prior information. The voxel value,
  \f$\lambda_j\f$, can then be described using the kernel matrix



  \f[
   \lambda_j=  \sum_{l=1}^L \alpha_l k_{jl}
  \f]

  where \f$k_{jl}\f$ is the \f$jl^{th}\f$ kernel element of the matrix, \f$\boldsymbol{K}\f$.
  The resulting algorithm is the following:

  \f[
  \alpha^{(n+1)}_j =  \frac{ \alpha^{(n)}_j }{\sum_{m} k^{(n)}_{jm} \sum_i p_{mi}} \sum_{m}k^{(n)}_{jm}\sum_i p_{mi}\frac{ y_i }{\sum_{q} p_{iq} \sum_l k^{(n)}_{ql}\alpha^{(n)}_l  + s_i}
  \f[

  where the  element, $\f jl \f$, of the kernel can be written as:

  \f[
    k^{(n)}_{jl} = k_m(\boldsymbol{v}_j,\boldsymbol{v}_l) \cdot k_p(\boldsymbol{z}^{(n)}_j,\boldsymbol{z}^{(n)}_l);
  \f]

  with

  \f[
   k_m(\boldsymbol{v}_j,\boldsymbol{v}_l) = \exp \left(\tiny - \frac{\|  \boldsymbol{v}_j-\boldsymbol{v}_l \|^2}{2 \sigma_m^2} \right) \exp \left(- \frac{\tiny \|  \boldsymbol{x}_j-\boldsymbol{x}_l \|^2}{ \tiny 2 \sigma_{dm}^2} \right)
  \f]

  being the MR component of the kernel and

  \f[
   k_p(\boldsymbol{z}^{(n)}_j,\boldsymbol{z}^{(n)}_l) = \exp \left(\tiny - \frac{\|  \boldsymbol{z}^{(n)}_j-\boldsymbol{z}^{(n)}_l \|^2}{2 \sigma_p^2} \right) \exp \left(\tiny - \frac{\|  \boldsymbol{x}_j-\boldsymbol{x}_l \|^2}{ \tiny{2 \sigma_{dp}^2}} \right)
  \f]

  is the part coming from the PET iterative update. Here, the Gaussian kernel functions have been modulated by the distance between voxels in the image space.

  \par Parameters for parsing

  \verbatim
  KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

  ; emission projection data
  input file :=

  hybrid:=1
  sigma_m:= 1                             ;is the parameter $\f \sigma_{m} \f$;
  sigma_p:=1                           ;is the parameter $\f \sigma_{p} \f$;
  sigma_dm:=1                          ;is the parameter $\f \sigma_{dm} \f$;
  sigma_dp:=1                           ;is the parameter $\f \sigma_{dp} \f$;
  neighbours_num:= 3                         ;is the cubic root of the number of voxels in the neighbourhood;
  anatomical_image_filename:=filename        ;is the filename of the anatomical image;
  num_non_zero_feat_elements:=1              ;is the number of non zero elements in the feature vector;
  only_2D:=0                                 ;=1 if you want to reconstruct 2D images;

  kernelised output filename prefix := kOUTPUTprefix ;this is  the name prefix for the reconstructed image after applying the kernel the reconstructed $\f \alpha \f$ coefficient.


  maximum absolute segment number to process :=
  zero end planes of segment 0 :=

  ; see ProjectorByBinPair hierarchy for possible values
  Projector pair type :=

  ; reserved value: 0 means none
  ; see KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData
  ; class documentation
  additive sinogram :=

  ; normalisation (and attenuation correction)
  ; time info can be used for dead-time correction

  ; see TimeFrameDefinitions
  time frame definition filename :=
  time frame number :=
  ; see BinNormalisation hierarchy for possible values
  Bin Normalisation type :=

  End KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters :=
  \endverbatim
*/

template <typename TargetT>
class KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData:
public  RegisteredParsingObject<KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
{

 private:
  typedef RegisteredParsingObject<KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
    base_type;

public:

  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name;

  //
  /*! \name Variables for STIR_MPI

  Only used when STIR_MPI is enabled.
  \todo move to protected area
  */
  //@{
  //! points to the information object needed to support distributed caching
  DistributedCachingInformation * caching_info_ptr;
  //#ifdef STIR_MPI
  //!enable/disable key for distributed caching
  bool distributed_cache_enabled;
  bool distributed_tests_enabled;
  bool message_timings_enabled;
  double message_timings_threshold;
  bool rpc_timings_enabled;
  //#endif
  //@}



  std::string sens_filenames;
  int subiter_num;


  //! Default constructor calls set_defaults()
  KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  //! Destructor
  /*! Calls end_distributable_computation()
   */
  ~KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  virtual TargetT *
    construct_target_ptr() const;

  /*! \name Functions to get parameters
   \warning Be careful with changing shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{

  const ProjData& get_proj_data() const;
  const shared_ptr<ProjData>& get_proj_data_sptr() const;
  const int get_max_segment_num_to_process() const;
  const bool get_zero_seg0_end_planes() const;
  const ProjData& get_additive_proj_data() const;
  const shared_ptr<ProjData>& get_additive_proj_data_sptr() const;
  const ProjectorByBinPair& get_projector_pair() const;
  const shared_ptr<ProjectorByBinPair>& get_projector_pair_sptr() const;
  const int get_time_frame_num() const;

  //kernel
  const std::string get_anatomical1_filename() const;
  const int get_neighbours_num() const;
  const int get_num_non_zero_feat() const;
  const double get_sigma_m() const;
  double get_sigma_p();
  double get_sigma_dp();
  double get_sigma_dm();
  const bool get_only_2D() const;
  int get_subiter_num();
  double get_kSD();
  bool get_hybrid();

   shared_ptr<TargetT>& get_kpnorm_sptr();
   shared_ptr<TargetT>& get_kmnorm_sptr();
   shared_ptr<TargetT>& get_anatomical1_sptr();
   const TimeFrameDefinitions& get_time_frame_definitions() const;
  const BinNormalisation& get_normalisation() const;
  const shared_ptr<BinNormalisation>& get_normalisation_sptr() const;
  //@}
  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with changing shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.

  */
  //@{

  int set_num_subsets(const int num_subsets);
  void set_subiter_num(int subiter_num);
  void set_kSD(double kSD);
  void set_kpnorm_sptr(shared_ptr<TargetT>&);
  void set_kmnorm_sptr(shared_ptr<TargetT>&);
  void set_anatomical1_sptr(shared_ptr<TargetT>&);
  void set_proj_data_sptr(const shared_ptr<ProjData>&);
  void set_max_segment_num_to_process(const int);
  void set_zero_seg0_end_planes(const bool);
  //N.E. Changed to ExamData
  virtual void set_additive_proj_data_sptr(const shared_ptr<ExamData>&);
  void set_projector_pair_sptr(const shared_ptr<ProjectorByBinPair>&) ;
  void set_frame_num(const int);
  void set_frame_definitions(const TimeFrameDefinitions&);
  virtual void set_normalisation_sptr(const shared_ptr<BinNormalisation>&);





  virtual void set_input_data(const shared_ptr<ExamData> &);
  //@}

  virtual void
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,
                                                          const TargetT &current_estimate,
                                                          const int subset_num);

#if 0
  // currently not used
  float sum_projection_data() const;
#endif
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

 protected:
  virtual Succeeded
    set_up_before_sensitivity(shared_ptr <TargetT > const& target_sptr);

  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num);

  /*!
    The Hessian (without penalty) is approximatelly given by:
    \f[ H_{jk} = - \sum_i P_{ij} h_i^{''}(y_i) P_{ik} \f]
    where
    \f[ h_i(l) = y_i log (l) - l; h_i^{''}(y_i) ~= -1/y_i; \f]
    and \f$P_{ij} \f$ is the probability matrix.
    Hence
    \f[ H_{jk} =  \sum_i P_{ij}(1/y_i) P_{ik} \f]

    In the above, we've used the plug-in approximation by replacing
    forward projection of the true image by the measured data. However, the
    later are noisy and this can create problems.

    \todo Two work-arounds for the noisy estimate of the Hessian are listed below,
    but they are currently not implemented.

    One could smooth the data before performing the quotient. This should be done
    after normalisation to avoid problems with the high-frequency components in
    the normalisation factors:
    \f[ H_{jk} =  \sum_i G_{ij}{1 \over n_i \mathrm{smooth}( n_i y_i)} G_{ik} \f]
    where the probability matrix is factorised in a detection efficiency part (i.e. the
    normalisation factors \f$n_i\f$) times a geometric part:
    \f[ P_{ij} = {1 \over n_i } G_{ij}\f]

    It has also been suggested to use \f$1 \over y_i+1 \f$ (at least if the data are still Poisson.
  */
  virtual Succeeded
      actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                             const TargetT& input,
                                                                             const int subset_num) const;

protected:
  //! Filename with input projection data
  std::string input_filename,kernelised_output_filename_prefix;
  std::string current_kimage_filename;

  //! Anatomical image filename
 std::string anatomical1_image_filename;
  mutable Array<3,float> distance;
  double kSt_dev;
  shared_ptr<TargetT> anatomical1_sptr;
  shared_ptr<TargetT> kpnorm_sptr,kmnorm_sptr;
 //kernel parameters
  int neighbours_num,num_non_zero_feat,num_elem_neighbourhood,num_voxels,dimz,dimy,dimx;
  double sigma_m;
  bool only_2D;
  bool hybrid;
  double sigma_p;
  double sigma_dp, sigma_dm;

  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_sptr;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

  /**********************/
  // image stuff
  // TODO to be replaced with single class or so (TargetT obviously)
  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy; // KT 10122001 appended _xy

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; // KT 10122001 new

  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  // KT 20/06/2001 new
  //! offset in the z-direction
  double Zoffset;
  /********************************/

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_ptr;

  //! signals whether to zero the data in the end planes of the projection data
  bool zero_seg0_end_planes;

  //! name of file in which additive projection data are stored
  std::string additive_projection_data_filename;


  shared_ptr<ProjData> additive_proj_data_sptr;

  shared_ptr<BinNormalisation> normalisation_sptr;

 // TODO doc
  int frame_num;
  std::string frame_definition_filename;
  TimeFrameDefinitions frame_defs;

//Loglikelihood computation parameters
 // TODO rename and move higher up in the hierarchy
  //! subiteration interval at which the loglikelihood function is evaluated
  int loglikelihood_computation_interval;

  //! indicates whether to evaluate the loglikelihood function for all bins or the current subset
  bool compute_total_loglikelihood;

  //! name of file in which loglikelihood measurements are stored
  std::string loglikelihood_data_filename;

  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys for parsing
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();
  //! checks values after parsing
  /*! Has to be called by post_processing in the leaf-class */
  virtual bool post_processing();

  //! Checks of the current subset scheme is approximately balanced
  /*! For this class, this means that the sub-sensitivities are
      approximately the same. The test simply looks at the number
      of views etc. It ignores unbalancing caused by normalisation_sptr
      (e.g. for instance when using asymmetric attenuation).
  */
  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;
 private:
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr;

  void add_view_seg_to_sensitivity(TargetT& sensitivity, const ViewSegmentNumbers& view_seg_nums) const;
  friend void RPC_process_related_viewgrams_gradient();

/*! Create a matrix containing the norm of the difference between two feature vectors, \f$ \|  \boldsymbol{z}^{(n)}_j-\boldsymbol{z}^{(n)}_l \| \f$. */
/*! This is done for the PET image which keeps changing*/
  void  calculate_norm_matrix(TargetT &normp,
                              const int &dimf_row,
                              int &dimf_col,
                              const TargetT& pet,
                              Array<3,float> distance);

/*! Create a matrix similarly to calculate_norm_matrix() but this is done for the anatomical image, */
/*! which does not  change over iteration.*/
  void  calculate_norm_const_matrix(TargetT &normm,
                              const int &dimf_row,
                              int &dimf_col);

/*! Estimate the SD of the anatomical image to be used as normalisation for the feature vector */
  void estimate_stand_dev_for_anatomical_image(double &SD);

/*! Compute for each voxel, jl, of the PET image the linear combination between the coefficient \f$ \alpha_{jl} \f$ and the kernel matrix \f$ k_{jl} \f$\f$ */
/*! The information is stored in the image, kImage */
  void compute_kernelised_image(TargetT &kImage, TargetT &Image, double sigma_m,
                                                            int neighbours_num,
                                                            const TargetT &current_estimate,
                                                            bool only_2D);

 /*! Similar to compute_kernelised_image() but this is the special case when the feature vectors contains only one non-zero element. */
 /*! The computation becomes faster because we do not need to create norm matrixes*/
void fast_compute_kernelised_image(TargetT &kImage, TargetT &Image, double sigma_m,
                                                          int neighbours_num,
                                                          const TargetT &current_estimate,
                                                          bool only_2D);

};

#ifdef STIR_MPI
//made available to be called from DistributedWorker object
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_gradient;
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_accumulate_loglikelihood;
#endif

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
