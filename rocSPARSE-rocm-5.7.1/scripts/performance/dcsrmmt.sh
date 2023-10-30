#!/usr/bin/env bash
# Author: Nico Trost

# Helper function
function display_help()
{
    echo "rocSPARSE benchmark helper script"
    echo "    [-h|--help] prints this help message"
    echo "    [-d|--device] select device"
    echo "    [-p|--path] path to rocsparse-bench"
    echo "    [-n|--sizen] number of dense columns"
}

# Check if getopt command is installed
type getopt > /dev/null
if [[ $? -ne 0 ]]; then
    echo "This script uses getopt to parse arguments; try installing the util-linux package";
    exit 1;
fi

dev=0
path=../../build/release/clients/staging
sizen=7

# Parse command line parameters
getopt -T
if [[ $? -eq 4 ]]; then
    GETOPT_PARSE=$(getopt --name "${0}" --longoptions help,device:,path:,sizen: --options hd:p:n: -- "$@")
else
    echo "Need a new version of getopt"
    exit 1
fi

if [[ $? -ne 0 ]]; then
    echo "getopt invocation failed; could not parse the command line";
    exit 1
fi

eval set -- "${GETOPT_PARSE}"

while true; do
    case "${1}" in
        -h|--help)
            display_help
            exit 0
            ;;
        -d|--device)
            dev=${2}
            shift 2 ;;
        -p|--path)
            path=${2}
            shift 2 ;;
        -n|--sizen)
            sizen=${2}
            shift 2 ;;
        --) shift ; break ;;
        *)  echo "Unexpected command line parameter received; aborting";
            exit 1
            ;;
    esac
done

bench=$path/rocsparse-bench

# Check if binary is available
if [ ! -f $bench ]; then
    echo $bench not found, exit...
    exit 1
else
    echo ">>" $(realpath $(ldd $bench | grep rocsparse | awk '{print $3;}'))
fi

# Generate logfile name
logname=dcsrmm_$(date +'%Y%m%d%H%M%S').log
truncate -s 0 $logname

# Run csrmm for all matrices available
for filename in ./matrices/*.csr; do
    $bench -f csrmm --precision d --device $dev --sizen $sizen --order 0 --transposeB T --alpha 1 --beta 0 --iters 200 --rocalution $filename 2>&1 | tee -a $logname
done
