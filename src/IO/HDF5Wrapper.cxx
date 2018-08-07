#include "stir/IO/HDF5Wrapper.h"

START_NAMESPACE_STIR

bool HDF5Wrapper::check_GE_signature(const std::string filename)
{
    H5::H5File file;
    file.openFile( filename, H5F_ACC_RDONLY );
    H5::StrType  vlst(0,37);  // 37 here is the length of the string (I got it from the text file generated by list2txt with the LIST000_decomp.BLF

    std::string read_str_scanner;
    std::string read_str_manufacturer;

    H5::DataSet dataset = file.openDataSet("/HeaderData/ExamData/scannerDesc");
    dataset.read(read_str_scanner,vlst);

    H5::DataSet dataset2= file.openDataSet("/HeaderData/ExamData/manufacturer");
    dataset2.read(read_str_manufacturer,vlst);

    if(read_str_scanner == "SIGNA PET/MR" &&
            read_str_manufacturer == "GE MEDICAL SYSTEMS")
        return true;

    return false;
}

HDF5Wrapper::HDF5Wrapper()
{

}

HDF5Wrapper::HDF5Wrapper(const std::string& filename)
{
    open(filename);

}

shared_ptr<Scanner>
HDF5Wrapper::get_scanner_sptr() const
{
    return this->scanner_sptr;
}

shared_ptr<ExamInfo>
HDF5Wrapper::get_exam_info_sptr() const
{
//    return this->exam_info_sptr;
}

H5::DataSet* HDF5Wrapper::get_listmode_data_ptr() const
{
    return dataset_list_sptr.get();
}

int HDF5Wrapper::get_listmode_size() const
{
    return m_list_size;
}

Succeeded
HDF5Wrapper::open(const std::string& filename)
{
    file.openFile( filename, H5F_ACC_RDONLY );

    if(HDF5Wrapper::check_GE_signature(filename))
    {
        warning("CListModeDataGESigna: "
                "Probably this is GESigna, but couldn't find scan start time etc."
                "The scanner is initialised from library instread from HDF5 header.");
        is_signa = true;

        this->scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));

        // TODO: read scanner type from the dataset: "/HeaderData/ExamData/scannerDesc"
        //this->exam_info_sptr.reset(new ExamInfo);

        return Succeeded::yes;
    }
    else
    {
        // Read from HDF5 header ...
        return initialise_scanner_from_HDF5();
    }
}

Succeeded HDF5Wrapper::initialise_scanner_from_HDF5()
{
    H5::DataSet str_num_rings = file.openDataSet("/HeaderData/SystemGeometry/");

    H5::DataSet str_inner_ring_diameter = file.openDataSet("/HeaderData/SystemGeometry/effectiveRingDiameter");

    H5::DataSet str_axial_blocks_per_module = file.openDataSet("/HeaderData/SystemGeometry/axialBlocksPerModule");
    H5::DataSet str_radial_blocks_per_module = file.openDataSet("/HeaderData/SystemGeometry/radialBlocksPerModule");

    H5::DataSet str_axial_blocks_per_unit = file.openDataSet("/HeaderData/SystemGeometry/axialBlocksPerUnit");
    H5::DataSet str_radial_blocks_per_unit = file.openDataSet("/HeaderData/SystemGeometry/radialBlocksPerUnit");

    H5::DataSet str_axial_units_per_module = file.openDataSet("/HeaderData/SystemGeometry/axialUnitsPerModule");
    H5::DataSet str_radial_units_per_module = file.openDataSet("/HeaderData/SystemGeometry/radialUnitsPerModule");

    H5::DataSet str_axial_modules_per_system = file.openDataSet("/HeaderData/SystemGeometry/axialModulesPerSystem");
    H5::DataSet str_radial_modules_per_system = file.openDataSet("/HeaderData/SystemGeometry/radialModulesPerSystem");

    //! \todo P.W: Find the crystal gaps and other info missing.

    //! \todo Convert to numbers.

//    scanner_sptr.reset(new Scanner(
//                           ));


    return Succeeded::yes;
}

Succeeded HDF5Wrapper::initialise_listmode_data(const std::string &path)
{
    if(path.size() == 0)
    {
        if(is_signa)
        {
            m_listmode_address = "/ListData/listData";
            m_size_of_record_signature = 4;
            m_max_size_of_record = 16;
        }
        else
            return Succeeded::no;
    }
    else
        m_listmode_address = path;

    dataset_list_sptr.reset(new H5::DataSet(file.openDataSet(m_listmode_address)));

    m_dataspace = dataset_list_sptr->getSpace();
    int dataset_list_Ndims = m_dataspace.getSimpleExtentNdims();

    hsize_t dims_out[dataset_list_Ndims];
    m_dataspace.getSimpleExtentDims( dims_out, NULL);

    const long long unsigned int tmp_size_of_record_signature = m_size_of_record_signature;
    m_memspace_ptr = new H5::DataSpace( dataset_list_Ndims,
                            &tmp_size_of_record_signature);


    return Succeeded::yes;
}


void HDF5Wrapper::get_next_listmode(hsize_t& current_offset, shared_ptr<char>& data_sptr)
{

    m_dataspace.selectHyperslab( H5S_SELECT_SET, &m_size_of_record_signature, &current_offset );
    dataset_list_sptr->read( data_sptr.get(), H5::PredType::STD_U8LE, *m_memspace_ptr, m_dataspace );
    current_offset += m_size_of_record_signature;

    ////  if (dataset_sptr->gcount()<static_cast<std::streamsize>(this->size_of_record_signature))
    ////    return Succeeded::no;
    //  const std::size_t size_of_record =data_sptr
    //	record.size_of_record_at_ptr(data_ptr, this->size_of_record_signature, false);
    //  assert(size_of_record <= this->max_size_of_record);
    //  if (size_of_record > this->size_of_record_signature)
    //  {
    //    offset[0] = current_offset;
    //    count[0]= size_of_record - this->size_of_record_signature;
    //    H5::DataSpace memspace( rank, &count[0] );
    //    dataspace.selectHyperslab( H5S_SELECT_SET, count, offset );
    //    dataset_sptr->read( data_ptr + this->size_of_record_signature,
    //                        H5::PredType::STD_U8LE, memspace, dataspace );
    //    current_offset += count[0];
    //  }
    //  // TODO error checking


}

//stir::Array<2, float>* data_sptr;
//shared_ptr<char> HDF5Wrapper::get_next_viewgram()
//{

//}

//stir::Array<2, float>* data_sptr;
//shared_ptr<char> HDF5Wrapper::get_next_sinogram()
//{

//}

//stir::Array<3, float>* data_sptr;
//shared_ptr<char> HDF5Wrapper::get_next_segment_by_sinogram()
//{

//}

//stir::Array<3, float>* data_sptr;
//shared_ptr<char> HDF5Wrapper::get_next_segment_by_viewgram()
//{

//}

// initialise_singles
// Can be used for normalisation, too.
// Can be used for nonTOF sinograms.
//Succeeded HDF5Wrapper::initialise_list_2D_arrays(const std::string &path)
//{
//    if(path.size() == 0)
//    {
//        if(is_signa)
//        {
//            m_listmode_address = "/ListData/listData";
//        }
//        else
//            return Succeeded::no;
//    }
//    else
//        m_listmode_address = path;

//    dataset_list_sptr.reset(new H5::DataSet(file.openDataSet(m_listmode_address)));

//    H5::DataSpace dataspace = dataset_list_sptr->getSpace();
//    dataset_list_Ndims = dataspace.getSimpleExtentNdims() ;


//    return Succeeded::yes;
//}



END_NAMESPACE_STIR

