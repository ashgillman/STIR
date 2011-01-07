#
# $Id$
#
set(dir buildblock)

set (dir_LIB_SOURCES ${dir}_LIB_SOURCES)

set(${dir_LIB_SOURCES}
  Array  
  IndexRange 
  ProjData 
  ProjDataInfo 
  ProjDataInfoCylindrical 
  ProjDataInfoCylindricalArcCorr 
  ProjDataInfoCylindricalNoArcCorr 
  ArcCorrection 
  ProjDataFromStream 
  ProjDataGEAdvance 
  ProjDataInMemory 
  ProjDataInterfile 
  Scanner 
  SegmentBySinogram 
  Segment 
  SegmentByView 
  Viewgram 
  Sinogram 
  RelatedViewgrams 
  scale_sinograms 
  interpolate_projdata 
  extend_projdata 
  DiscretisedDensity 
  VoxelsOnCartesianGrid 
  utilities 
  interfile_keyword_functions 
  zoom 
  NumericType ByteOrder 
  KeyParser  
  recon_array_functions 
  linear_regression overlap_interpolate 
  error warning  
  DataSymmetriesForViewSegmentNumbers 
  TimeFrameDefinitions 
  ParsingObject 
	ArrayFilter1DUsingConvolutionSymmetricKernel 
	ArrayFilterUsingRealDFTWithPadding 
	SeparableArrayFunctionObject 
	SeparableMetzArrayFilter 
	MedianArrayFilter3D 
	MedianImageFilter3D 
	MinimalArrayFilter3D 
	MinimalImageFilter3D 
	SeparableCartesianMetzImageFilter 
	TruncateToCylindricalFOVImageProcessor 
	ThresholdMinToSmallPositiveValueDataProcessor 
	ChainedDataProcessor 
	ArrayFilter1DUsingConvolution 
	SeparableConvolutionImageFilter 
	NonseparableConvolutionUsingRealDFTImageFilter 
	SSRB 
	inverse_SSRB 
	centre_of_gravity 
	DynamicDiscretisedDensity 
	DynamicProjData 
	MultipleProjData 
	GatedProjData 
	ArrayFilter2DUsingConvolution 
	ArrayFilter3DUsingConvolution 
	find_fwhm_in_image
)

check_function_exists(getopt HAVE_SYSTEM_GETOPT)
if (HAVE_SYSTEM_GETOPT)
  add_definitions(-D HAVE_SYSTEM_GETOPT)
else()
  # add our own version of getopt to buildblock
  list(APPEND ${dir_LIB_SOURCES} getopt)
endif()

add_library(${dir} ${${dir_LIB_SOURCES}}    )