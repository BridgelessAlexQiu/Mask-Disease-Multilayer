#!/bin/bash
#SBATCH --time=7:00:00
#SBATCH -p standard
#SBATCH -n 1
#SBATCH --mem=98000
#SBATCH -A nssac_students
#SBATCH -o simu1.out
#SBATCH -e simu1.error

module load anaconda/2019.10-py3.7

python3 generate_network.py 10
python3 generate_network.py 20
python3 generate_network.py 30
python3 generate_network.py 40
python3 generate_network.py 50
python3 generate_network.py 60
python3 generate_network.py 70
python3 generate_network.py 80
python3 generate_network.py 90
python3 generate_network.py 100
