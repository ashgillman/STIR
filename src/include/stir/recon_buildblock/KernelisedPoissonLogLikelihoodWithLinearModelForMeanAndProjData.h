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

#ifndef __stir_recon_buildblock_KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__
#define __stir_recon_buildblock_KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/Array.h"
#include "stir/shared_ptr.h"
#include <string>
START_NAMESPACE_STIR

class DistributedCachingInformation;



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
  the sinogram-based objective function. Each voxel value of the image, \f$ \boldsymbol{\lambda}\f$, can be represented as a
  linear combination using the kernel method. % If we have an image with prior information, we can construct for each voxel
  \f$ j \f$ of the PET image a feature vector, $\f \boldsymbol{v}_j \f$, using the prior information. The voxel value,
  \f$\lambda_j\f$, can then be described using the kernel matrix



  \f[
   \lambda_j=  \sum_{l=1}^L \alpha_l k_{jl}
  \f]

  where \f$k_{jl}\f$ is the \f$jl^{th}\f$ kernel element of the matrix, \f$\boldsymbol{K}\f$.
  The resulting algorithm with OSEM, for example, is the following:

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

  hybrid:=1
  sigma m:= 1                                ;is the parameter $\f \sigma_{m} \f$;
  sigma p:=1                                 ;is the parameter $\f \sigma_{p} \f$;
  sigma dm:=1                                ;is the parameter $\f \sigma_{dm} \f$;
  sigma dp:=1                                ;is the parameter $\f \sigma_{dp} \f$;
  number of neighbours:= 3                   ;is the cubic root of the number of voxels in the neighbourhood;
  anatomical image filename:=filename       ;is the filename of the anatomical image;
  number of non-zero feature elements:=1     ;is the number of non zero elements in the feature vector;
  only_2D:=0                                 ;=1 if you want to reconstruct 2D images;

  kernelised output filename prefix := kOUTPUTprefix ;this is  the name prefix for the reconstructed image after applying the kernel the reconstructed $\f \alpha \f$ coefficient.


  End KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters :=
  \endverbatim
*/

template <typename TargetT>
class KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData:
public  RegisteredParsingObject<KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT> >
{

 private:
  typedef RegisteredParsingObject<KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT> >
    base_type;

public:

  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name;


  //! Default constructor calls set_defaults()
  KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  //! Destructor
  /*! Calls end_distributable_computation()
   */
  ~KernelisedPoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  /*! \name Functions to get parameters
   \warning Be careful with changing shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{

  //kernel
  const std::string get_anatomical_filename() const;
  const int get_num_neighbours() const;
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
   shared_ptr<TargetT>& get_anatomical_sptr();

  //@}
  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with changing shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.

  */
  //@{

  void set_subiter_num(int subiter_num);
  void set_kSD(double kSD);
  void set_kpnorm_sptr(shared_ptr<TargetT>&);
  void set_kmnorm_sptr(shared_ptr<TargetT>&);
  void set_anatomical_sptr(shared_ptr<TargetT>&);

  //@}

  virtual void
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,
                                                          const TargetT &current_estimate,
                                                          const int subset_num);
protected:

  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num);

protected:
  //! Filename with input projection data
  std::string input_filename,kernelised_output_filename_prefix;
  std::string current_kimage_filename;
  std::string sens_filenames;
 int subiter_num;

  //! Anatomical image filename
 std::string anatomical_image_filename;
  mutable Array<3,float> distance;
  double kSt_dev;
  shared_ptr<TargetT> anatomical_sptr;
  shared_ptr<TargetT> kpnorm_sptr,kmnorm_sptr;
 //kernel parameters
  int num_neighbours,num_non_zero_feat,num_elem_neighbourhood,num_voxels,dimz,dimy,dimx;
  double sigma_m;
  bool only_2D;
  bool hybrid;
  double sigma_p;
  double sigma_dp, sigma_dm;


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

 private:

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
  void compute_kernelised_image(TargetT &kImage, TargetT &Image,
                                                            const TargetT &current_estimate);

 /*! Similar to compute_kernelised_image() but this is the special case when the feature vectors contains only one non-zero element. */
 /*! The computation becomes faster because we do not need to create norm matrixes*/
void fast_compute_kernelised_image(TargetT &kImage, TargetT &Image,
                                                          const TargetT &current_estimate);

};


END_NAMESPACE_STIR

#endif
