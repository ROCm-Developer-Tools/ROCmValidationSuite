# ROCmValidationSuite
The ROCm Validation Suite (RVS) is a system administrator’s and cluster manager's tool for detecting and troubleshooting common problems affecting AMD GPU(s) running in a high-performance computing environment, enabled using the ROCm software stack on a compatible platform.

The RVS is a collection of tests, benchmarks and qualification tools each targeting a specific sub-system of the ROCm platform. All of the tools are implemented in software and share a common command line interface. Each set of tests are implemented in a “module” which is a library encapsulating the functionality specific to the tool. The CLI can specify the directory containing modules to use when searching for libraries to load. Each module may have a set of options that it defines and a configuration file that supports its execution.

List of tests that can be run by a customer/user [link](./CUSTOMER.md).

The function of each module see this [link](./FEATURES.md).

Examples and about config files [link](./doc/ugsrc/ug1main.md).

## Build and Installation

## Prerequisites 
Please do this before compilation/installing compiled package.

Ubuntu : 
      
        sudo apt-get -y update && sudo apt-get install -y libpci3 libpci-dev doxygen unzip cmake git libpciaccess-dev

 CentOS7.x: 
        
        sudo yum install -y cmake3 doxygen pciutils-devel rpm rpm-build git gcc-c++ libpciaccess-devel
 
 RHEL7.x : 
        
       sudo yum install -y cmake3 doxygen rpm rpm-build git gcc-c++ libpciaccess-devel
        
       wget http://mirror.centos.org/centos/7/os/x86_64/Packages/pciutils-devel-3.5.1-3.el7.x86_64.rpm
        
       sudo rpm -ivh pciutils-devel-3.5.1-3.el7.x86_64.rpm
		
 SLES :  
		    
       sudo SUSEConnect -p sle-module-desktop-applications/15.1/x86_64
       
       sudo SUSEConnect --product sle-module-development-tools/15.1/x86_64
       
       sudo zypper  install -y cmake doxygen pciutils-devel libpci3 rpm git rpm-build gcc-c++ libpciaccess-devel

## Install ROCm stack, rocblas and rocm-smi-lib64
Install ROCm stack for Ubuntu/CentOS/SLES/RHEL, Refer https://github.com/RadeonOpenCompute/ROCm

_**Note:**_

rocm_smi64 package name changed to rocm-smi-lib64 from ROCm3.0 onwards. If you are using ROCm release < 3.0 , install the package as "rocm_smi64".
 
Install rocBLAS and rocm-smi-lib64 : 

   Ubuntu : 
   
           sudo apt-get install rocblas rocm-smi-lib64
   
   CentOS & RHEL : 
            
           sudo yum install --nogpgcheck rocblas rocm-smi-lib64
   
   SUSE : 
         
           sudo zypper install rocblas rocm-smi-lib64

_**Note:**_
If  rocm-smi-lib64 is already installed but "/opt/rocm/rocm_smi/ path doesn't exist. Do below:

Ubuntu : sudo dpkg -r rocm-smi-lib64 && sudo apt install rocm-smi-lib64

CentOS & RHEL : sudo rpm -e  rocm-smi-lib64 && sudo yum install  rocm-smi-lib64

SUSE : sudo rpm -e  rocm-smi-lib64 && sudo zypper install  rocm-smi-lib64

## Building from Source
This section explains how to get and compile current development stream of RVS.

### Clone repository
    git clone https://github.com/ROCm-Developer-Tools/ROCmValidationSuite.git

### Configure and build RVS:

    cd ROCmValidationSuite
 If OS is Ubuntu and SLES, use cmake
    
     cmake  -DROCM_PATH=<rocm_installed_path> -DCMAKE_INSTALL_PREFIX=<rocm_installed_path> -DCMAKE_PACKAGING_INSTALL_PREFIX=<rocm_installed_path> ./ -B./build
     
     eg/-
     cmake -DROCM_PATH=/opt/rocm-4.0.0-6220 -DCMAKE_INSTALL_PREFIX=/opt/rocm-4.0.0-6220 -DCMAKE_PACKAGING_INSTALL_PREFIX=/opt/rocm-4.0.0-6220 ./ -B./build
     
     make -C ./build
     
If OS is CentOS and RHEL, use cmake3

    cmake3 ./ -B./build
 
    make -C ./build

### Build package:

     cd ./build
     
     make package

**Note:**_ based on your OS, only DEB or RPM package will be built. You may
ignore an error for the unrelated configuration

### Install package:

    Ubuntu : sudo dpkg -i rocm-validation-suite*.deb
    CentOS & RHEL & SUSE : sudo yum install rocm-validation-suite*.rpm

**Note:**
RVS is getting packaged as part of ROCm release starting from 3.0. You can install pre-compiled package as below.
Please make sure Prerequisites, ROCm stack, rocblas and rocm-smi-lib64 are already installed

    Ubuntu : sudo apt install rocm-validation-suite
    CentOS & RHEL : sudo yum install rocm-validation-suite
    SUSE : sudo zypper install rocm-validation-suite

## Running RVS

    cd ./build/bin
    sudo ./rvsqa.new.sh ; It will run complete rvs test suite

### Running version pre-complied and packaged with ROCm release
   
    sudo /opt/rocm/rvs/rvs -d 3
    sudo /opt/rocm/rvs/rvsqa.new.sh 
   
Similarly, you can run all tests as mentioned in "rvsqa.new.sh" script, present at "testscripts/rvsqa.new.sh"

## Regression

Simple regression has been implemented. You may find more about it
on this [link](./REGRESSION.md).
