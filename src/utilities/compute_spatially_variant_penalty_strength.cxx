/*
    Copyright (C) 2020, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the Lesser GNU General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    Lesser GNU General Public License for more details.

    See STIR/LICENSE.txt for details
*/

/*!
  \file
  \ingroup utilities

  \brief Computes the spatially variant penalty strength

  \par Usage
  \verbatim
     compute_spatially_variant_penalty_strength template_proj_data
  \endverbatim

  Computes a spatially variant penalty strength, either using:

  todo: add methods and documentation
  See Tsai, Y.-J., Schramm, G., Ahn, S., Bousse, A., Arridge, S., Nuyts, J., Hutton, B. F., Stearns, C. W.,
    & Thielemans, K. (2020). Benefits of Using a Spatially-Variant Penalty Strength With Anatomical Priors
    in PET Reconstruction. IEEE Transactions on Medical Imaging, 39(1), 11–22. https://doi.org/10.1109/TMI.2019.2913889
   for more details

  \author Robert Twyman
*/

#include <stir/info.h>
#include <stir/HighResWallClockTimer.h>
#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#ifndef STIR_NO_NAMESPACES
using std::endl;
using std::cerr;
#endif



START_NAMESPACE_STIR
    static void print_usage_and_exit()
    {
      //todo:update usage
      std::cerr<<"\nUsage: compute_spatially_variant_penalty_strength template_proj_data\n";
      exit(EXIT_FAILURE);
    }
END_NAMESPACE_STIR

USING_NAMESPACE_STIR
class KappaComputation: public ParsingObject
{
    //All methods need documenting
public:
    KappaComputation();
    void set_defaults();
    void process_data();
    typedef DiscretisedDensity<3,float> target_type;

protected:
    shared_ptr<DiscretisedDensity<3,float>> current_image_estimate_sptr; // need templating
    shared_ptr<DiscretisedDensity<3,float>> template_image_sptr; // need templating

    shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;
    shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > output_file_format_sptr;

    void compute_kappa_at_current_image_estimate();
    void compute_kappa_with_approximate();

private:
    std::string current_image_estimate_filename;
    std::string template_image_filename;
    std::string kappa_filename;
    void initialise_keymap();
    bool post_processing();

    void sqrt_image(DiscretisedDensity<3,float> &output_image_sptr);
};

KappaComputation::KappaComputation()
{
  set_defaults();
}

void
KappaComputation::set_defaults()
{
  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
  output_file_format_sptr = OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr();
}

void
KappaComputation::initialise_keymap()
{
  parser.add_start_key("Kappa Computation Parameters");
  parser.add_key("kappa filename", &kappa_filename);
  parser.add_key("current image estimate", &current_image_estimate_filename);
  parser.add_key("template image", &template_image_filename);
  parser.add_parsing_key("objective function type", &objective_function_sptr);
  parser.add_stop_key("End");
}

bool
KappaComputation::post_processing()
{
  if (is_null_ptr(this->objective_function_sptr))
  {
    error("objective_function_sptr is null");
    return true;
  }

  if (current_image_estimate_filename.empty() && template_image_filename.empty())
  {
    error("Requires either current_image_estimate_filename or template_image_filename");
    return true;
  }
  return false;
}

void
KappaComputation::process_data()
{
  if (!current_image_estimate_filename.empty())
  {
    current_image_estimate_sptr = read_from_file<DiscretisedDensity<3,float>>(current_image_estimate_filename);
    compute_kappa_at_current_image_estimate();
  }

  else if (!template_image_filename.empty())
  {
    template_image_sptr = read_from_file<DiscretisedDensity<3,float>>(template_image_filename);
    compute_kappa_with_approximate();
  }

  else
    error("process_data: Either both current_image_estimate_filename and template_image_filename are empty.");

  info("Spatially variant penalty strength (Kappa) has been computed and saved.");
}


void
KappaComputation::compute_kappa_at_current_image_estimate()
{
  info("Computing the spatially variant penalty strength at the current image estimate.");

  auto output_image_sptr = current_image_estimate_sptr->get_empty_copy();
  output_image_sptr->fill(0.);
  auto ones_image_sptr = current_image_estimate_sptr->get_empty_copy();
  ones_image_sptr->fill(1.);

  // Unfortunately we have to setup. This involves the computation of the sensitivity
  objective_function_sptr->set_up(current_image_estimate_sptr);
  objective_function_sptr->accumulate_Hessian_times_input(*output_image_sptr, *current_image_estimate_sptr, *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of accumulate_Hessian_times_input
  sqrt_image(*output_image_sptr);
  output_file_format_sptr->write_to_file(kappa_filename, *output_image_sptr);
}

void
KappaComputation::compute_kappa_with_approximate()
{

  info("Computing the spatially variant penalty strength using approximate hessian.");
  auto output_image_sptr = template_image_sptr->get_empty_copy();
  output_image_sptr->fill(0.);

  auto ones_image_sptr = template_image_sptr->get_empty_copy();
  ones_image_sptr->fill(1.);

  // Unfortunately we have to setup. This involves the computation of the sensitivity
  objective_function_sptr->set_up(template_image_sptr);

  // Approximate Hessian computation will error for a lot of priors so we ignore it!
  info("Priors do not have an approximation of the Hessian. Therefore we will ignore the prior.");
  objective_function_sptr->add_multiplication_with_approximate_Hessian_without_penalty(*output_image_sptr, *ones_image_sptr);

  // Kappa is defined as the sqrt of the output of add_multiplication_with_approximate_Hessian_without_penalty
  sqrt_image(*output_image_sptr);
  output_file_format_sptr->write_to_file(kappa_filename, *output_image_sptr);
}

void
KappaComputation::sqrt_image(DiscretisedDensity<3,float> &output_image_sptr)
{
  // Square root the output
  typename DiscretisedDensity<3,float>::const_full_iterator output_iter = output_image_sptr.begin_all_const();
  const typename DiscretisedDensity<3,float>::const_full_iterator end_prior_output_iter = output_image_sptr.end_all_const();
  typename DiscretisedDensity<3,float>::full_iterator tmp_iter = output_image_sptr.begin_all();
  while (output_iter!=end_prior_output_iter)
  {
    *tmp_iter = sqrt(*output_iter);
    ++tmp_iter; ++output_iter;
  }
}


int
main (int argc, char * argv[])
{

  KappaComputation kappa_computer;

  if (argc!=2)
    print_usage_and_exit();
  else
    kappa_computer.parse(argv[1]);

  kappa_computer.process_data();
  return 0;
}
