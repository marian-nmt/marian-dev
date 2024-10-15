//Copyright (C) 2011  Carl Rogers
//Released under MIT License
//license available in LICENSE file, or at http://www.opensource.org/licenses/mit-license.php

#ifndef LIBCNPY_H_
#define LIBCNPY_H_

#include "3rd_party/zlib/zlib.h"

#include<string>
#include<stdexcept>
#include<sstream>
#include<vector>
#include<cstdio>
#include<typeinfo>
#include<iostream>
#include<cassert>
#include<map>
#include <memory>

#ifdef __APPLE__
#include <unistd.h>
#endif

namespace cnpy {

    struct NpyArray {
        std::vector<char> bytes;
        std::vector<unsigned int> shape;
        
        // See cnpy::map_type() for a list of valid char codes and their mappings. 
        // Numpy seems to only understand five types {f, i, u, b, c} paired with
        // word_size.
        char type; 
        unsigned int word_size{1};
        
        bool fortran_order{0};

        NpyArray() {}

        void resize(size_t n) {
            return bytes.resize(n);
        }

        char* data() {
            return bytes.data();
        }

        const char* data() const {
            return bytes.data();
        }

        size_t size() {
            return bytes.size();
        }
    };

    typedef std::shared_ptr<NpyArray> NpyArrayPtr;
    typedef std::map<std::string, NpyArrayPtr> npz_t;

    char BigEndianTest();
    char map_type(const std::type_info& t);
    static inline std::vector<char> create_npy_header(char type, size_t word_size, const unsigned int* shape, const unsigned int ndims);
    template<typename T> std::vector<char> create_npy_header(const T* data, const unsigned int* shape, const unsigned int ndims);
    void parse_npy_header(FILE* fp, char& type, unsigned int& word_size, unsigned int*& shape, unsigned int& ndims, bool& fortran_order);
    void parse_zip_footer(FILE* fp, unsigned short& nrecs, unsigned int& global_header_size, unsigned int& global_header_offset);
    npz_t npz_load(std::string fname);
    NpyArrayPtr npz_load(std::string fname, std::string varname);
    NpyArrayPtr npy_load(std::string fname);

    template<typename T> std::vector<char>& operator+=(std::vector<char>& lhs, const T rhs) {
        //write in little endian
        for(char byte = 0; byte < sizeof(T); byte++) {
            char val = *((char*)&rhs+byte);
            lhs.push_back(val);
        }
        return lhs;
    }

    template<> std::vector<char>& operator+=(std::vector<char>& lhs, const std::string rhs);
    template<> std::vector<char>& operator+=(std::vector<char>& lhs, const char* rhs);


    template<typename T> std::string tostring(T i, int /*pad*/ = 0, char /*padval*/ = ' ') {
        std::stringstream s;
        s << i;
        return s.str();
    }

    template<typename T> void npy_save(std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w") {
        FILE* fp = NULL;

        if(mode == "a") fp = fopen(fname.c_str(),"r+b");

        if(fp) {
            //file exists. we need to append to it. read the header, modify the array size
            unsigned int word_size, tmp_dims;
            char type;
            unsigned int* tmp_shape = 0;
            bool fortran_order;
            parse_npy_header(fp,type,word_size,tmp_shape,tmp_dims,fortran_order);
            assert(!fortran_order);

            if(word_size != sizeof(T)) {
                std::cout<<"libnpy error: "<<fname<<" has word size "<<word_size<<" but npy_save appending data sized "<<sizeof(T)<<"\n";
                assert( word_size == sizeof(T) );
            }
            if(tmp_dims != ndims) {
                std::cout<<"libnpy error: npy_save attempting to append misdimensioned data to "<<fname<<"\n";
                assert(tmp_dims == ndims);
            }

            for(int i = 1; i < ndims; i++) {
                if(shape[i] != tmp_shape[i]) {
                    std::cout<<"libnpy error: npy_save attempting to append misshaped data to "<<fname<<"\n";
                    assert(shape[i] == tmp_shape[i]);
                }
            }
            tmp_shape[0] += shape[0];

            fseek(fp,0,SEEK_SET);
            std::vector<char> header = create_npy_header(data,tmp_shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
            fseek(fp,0,SEEK_END);

            delete[] tmp_shape;
        }
        else {
            fp = fopen(fname.c_str(),"wb");
            std::vector<char> header = create_npy_header(data,shape,ndims);
            fwrite(&header[0],sizeof(char),header.size(),fp);
        }

        unsigned int nels = 1;
        for(int i = 0;i < ndims;i++) nels *= shape[i];

        fwrite(data,sizeof(T),nels,fp);
        fclose(fp);
    }

    template<typename T> void npz_save(std::string zipname, std::string fname, const T* data, const unsigned int* shape, const unsigned int ndims, std::string mode = "w")
    {
        //first, append a .npy to the fname
        fname += ".npy";

        //now, on with the show
        FILE* fp = NULL;
        unsigned short nrecs = 0;
        unsigned int global_header_offset = 0;
        std::vector<char> global_header;

        if(mode == "a") fp = fopen(zipname.c_str(),"r+b");

        if(fp) {
            //zip file exists. we need to add a new npy file to it.
            //first read the footer. this gives us the offset and size of the global header
            //then read and store the global header.
            //below, we will write the the new data at the start of the global header then append the global header and footer below it
            unsigned int global_header_size;
            parse_zip_footer(fp,nrecs,global_header_size,global_header_offset);
            fseek(fp,global_header_offset,SEEK_SET);
            global_header.resize(global_header_size);
            size_t res = fread(&global_header[0],sizeof(char),global_header_size,fp);
            if(res != global_header_size){
                throw std::runtime_error("npz_save: header read error while adding to existing zip");
            }
            fseek(fp,global_header_offset,SEEK_SET);
        }
        else {
            fp = fopen(zipname.c_str(),"wb");
        }

        std::vector<char> npy_header = create_npy_header(data,shape,ndims);

        unsigned long nels = 1;
        for (int m=0; m<ndims; m++ ) nels *= shape[m];
        auto nbytes = nels*sizeof(T) + npy_header.size();

        //get the CRC of the data to be added
        unsigned int crc = crc32(0L,(unsigned char*)&npy_header[0],npy_header.size());
        crc = crc32(crc,(unsigned char*)data,nels*sizeof(T));

        //build the local header
        std::vector<char> local_header;
        local_header += "PK"; //first part of sig
        local_header += (unsigned short) 0x0403; //second part of sig
        local_header += (unsigned short) 20; //min version to extract
        local_header += (unsigned short) 0; //general purpose bit flag
        local_header += (unsigned short) 0; //compression method
        local_header += (unsigned short) 0; //file last mod time
        local_header += (unsigned short) 0;     //file last mod date
        local_header += (unsigned int) crc; //crc
        local_header += (unsigned int) nbytes; //compressed size
        local_header += (unsigned int) nbytes; //uncompressed size
        local_header += (unsigned short) fname.size(); //fname length
        local_header += (unsigned short) 0; //extra field length
        local_header += fname;

        //build global header
        global_header += "PK"; //first part of sig
        global_header += (unsigned short) 0x0201; //second part of sig
        global_header += (unsigned short) 20; //version made by
        global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
        global_header += (unsigned short) 0; //file comment length
        global_header += (unsigned short) 0; //disk number where file starts
        global_header += (unsigned short) 0; //internal file attributes
        global_header += (unsigned int) 0; //external file attributes
        global_header += (unsigned int) global_header_offset; //relative offset of local file header, since it begins where the global header used to begin
        global_header += fname;

        //build footer
        std::vector<char> footer;
        footer += "PK"; //first part of sig
        footer += (unsigned short) 0x0605; //second part of sig
        footer += (unsigned short) 0; //number of this disk
        footer += (unsigned short) 0; //disk where footer starts
        footer += (unsigned short) (nrecs+1); //number of records on this disk
        footer += (unsigned short) (nrecs+1); //total number of records
        footer += (unsigned int) global_header.size(); //nbytes of global headers
        footer += (unsigned int) (global_header_offset + nbytes + local_header.size()); //offset of start of global headers, since global header now starts after newly written array
        footer += (unsigned short) 0; //zip file comment length

        //write everything
        fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
        fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
        fwrite(data,sizeof(T),nels,fp);
        fwrite(&global_header[0],sizeof(char),global_header.size(),fp);
        fwrite(&footer[0],sizeof(char),footer.size(),fp);
        //BUGBUG: no check for write error
        fclose(fp);
    }

    //one item pass to npz_save() below
    struct NpzItem : public NpyArray
    {
        std::string name; //name of item in .npz file (without .npy)
        char type;        // type of item

        template<typename T>
        NpzItem(const std::string& name, const std::vector<T>& data, const std::vector<unsigned int>& dataShape) :
            name(name), type(map_type(typeid(T)))
        {
            shape = dataShape;
            word_size = sizeof(T);
            bytes.resize(data.size() * word_size);
            auto* p = (const char*)data.data();
            std::copy(p, p + bytes.size(), bytes.begin());
        }

        NpzItem(const std::string& name, const std::string& data, const std::vector<unsigned int>& dataShape) :
            name(name), type(map_type(typeid(char)))
        {
            shape = dataShape;
            word_size = sizeof(char);
            std::copy(data.data(), data.data() + data.size() + 1, bytes.begin());
        }

        NpzItem(const std::string& name,
                const std::vector<char>& data,
                const std::vector<unsigned int>& dataShape,
                char type_, size_t word_size_) :
            name(name), type(type_)
        {
            shape = dataShape;
            word_size = (unsigned int)word_size_;
            bytes.resize(data.size());
            std::copy(data.begin(), data.end(), bytes.begin());
        }
    };

    //same as npz_save() except that it saves multiple items to .npz file in a single go, which is required when writing to HDFS
    static inline
    void npz_save(std::string zipname, const std::vector<NpzItem>& items)
    {
        auto tmpname = zipname + "$$"; // TODO: add thread id or something
#ifndef __ANDROID__
        unlink(tmpname.c_str()); // when saving to HDFS, we cannot overwrite an existing file
#endif
        FILE* fp = fopen(tmpname.c_str(),"wb");
        if (!fp)
            throw std::runtime_error("npz_save: error opening file for writing: " + tmpname);

        std::vector<char> global_header;
        std::vector<char> local_header;
        for (const auto& item : items)
        {
            auto fname = item.name;
            //first, form a "file name" by appending .npy to the item's name
            fname += ".npy";

            const auto* data      = item.bytes.data();
            const auto* shape     = item.shape.data();
            const auto  type      = item.type;
            const auto  word_size = item.word_size;
            const unsigned int ndims = (unsigned int)item.shape.size();
            std::vector<char> npy_header = create_npy_header(type,word_size,shape,ndims);

            unsigned long nels = 1;
            for (size_t m=0; m<ndims; m++ ) nels *= shape[m];
            auto nbytes = nels*word_size + npy_header.size();

            //get the CRC of the data to be added
            unsigned int crc = crc32(0L,(unsigned char*)&npy_header[0],(uInt)npy_header.size());
            crc = crc32(crc,(unsigned char*)data,nels*word_size);

            //build the local header
            local_header.clear();
            local_header += "PK"; //first part of sig
            local_header += (unsigned short) 0x0403; //second part of sig
            local_header += (unsigned short) 20; //min version to extract
            local_header += (unsigned short) 0; //general purpose bit flag
            local_header += (unsigned short) 0; //compression method
            local_header += (unsigned short) 0; //file last mod time
            local_header += (unsigned short) 0;     //file last mod date
            local_header += (unsigned int) crc; //crc
            local_header += (unsigned int) nbytes; //compressed size
            local_header += (unsigned int) nbytes; //uncompressed size
            local_header += (unsigned short) fname.size(); //fname length
            local_header += (unsigned short) 0; //extra field length
            local_header += fname;

            //write everything
            unsigned int local_header_offset = ftell(fp); // this is where this local item will begin in the file. Tis gets stored in the corresponding global header.
            fwrite(&local_header[0],sizeof(char),local_header.size(),fp);
            fwrite(&npy_header[0],sizeof(char),npy_header.size(),fp);
            fwrite(data,word_size,nels,fp);

            // append to global header
            // A concatenation of global headers for all objects gets written to the end of the file.
            global_header += "PK"; //first part of sig
            global_header += (unsigned short) 0x0201; //second part of sig
            global_header += (unsigned short) 20; //version made by
            global_header.insert(global_header.end(),local_header.begin()+4,local_header.begin()+30);
            global_header += (unsigned short) 0; //file comment length
            global_header += (unsigned short) 0; //disk number where file starts
            global_header += (unsigned short) 0; //internal file attributes
            global_header += (unsigned int) 0; //external file attributes
            global_header += (unsigned int) local_header_offset; //relative offset of local file header, since it begins where the global header used to begin
            global_header += fname;
        }

        //write global headers
        unsigned int global_header_offset = ftell(fp); // this is where the global headers get written to in the file
        fwrite(&global_header[0],sizeof(char),global_header.size(),fp);

        //build footer
        auto nrecs = items.size();
        std::vector<char> footer;
        footer += "PK"; //first part of sig
        footer += (unsigned short) 0x0605; //second part of sig
        footer += (unsigned short) 0; //number of this disk
        footer += (unsigned short) 0; //disk where footer starts
        footer += (unsigned short) nrecs; //number of records on this disk
        footer += (unsigned short) nrecs; //total number of records
        footer += (unsigned int) global_header.size(); //nbytes of global headers
        footer += (unsigned int) global_header_offset; //offset of start of global headers
        footer += (unsigned short) 0; //zip file comment length

        //write footer
        fwrite(&footer[0],sizeof(char),footer.size(),fp);

        //close up
        fflush(fp);
        bool bad = ferror(fp) != 0;
        fclose(fp);

        // move to final location (atomically)
#ifdef _MSC_VER
        unlink(zipname.c_str()); // needed for Windows
#endif
        bad = bad || (rename(tmpname.c_str(), zipname.c_str()) == -1);

        if (bad)
        {
#ifndef __ANDROID__
            unlink(tmpname.c_str());
#endif
            throw std::runtime_error("npz_save: error saving to file: " + zipname);
        }
    }

    static inline
    std::vector<char> create_npy_header(char type, size_t word_size, const unsigned int* shape, const unsigned int ndims) {

        std::vector<char> dict;
        dict += "{'descr': '";
        dict += BigEndianTest();
        dict += type;
        dict += tostring(word_size);
        dict += "', 'fortran_order': False, 'shape': (";
        dict += tostring(shape[0]);
        for(size_t i = 1;i < ndims;i++) {
            dict += ", ";
            dict += tostring(shape[i]);
        }
        if(ndims == 1) dict += ",";
        dict += "), }";
        //pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
        int remainder = 16 - (10 + dict.size()) % 16;
        dict.insert(dict.end(),remainder,' ');
        dict.back() = '\n';

        std::vector<char> header;
        header += (char) (0x93 - 0x100);
        header += "NUMPY";
        header += (char) 0x01; //major version of numpy format
        header += (char) 0x00; //minor version of numpy format
        header += (unsigned short) dict.size();
        header.insert(header.end(),dict.begin(),dict.end());

        return header;
    }

    template<typename T> std::vector<char> create_npy_header(const T*, const unsigned int* shape, const unsigned int ndims) {
        return create_npy_header(map_type(typeid(T)), sizeof(T), shape, ndims);
    }
}

#endif
