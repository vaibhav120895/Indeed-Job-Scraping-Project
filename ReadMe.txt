Scraping Job Openings from Indeed for Analysis and personal convenience.

As I was moving close to my graduation, there was a need to search and apply for more and more job openings for the roles that I was looking for. 
In the process, I had to navigate to Indeed and search every-time I sat down for doing some applications. 

To automate this process, I wrote a Machine Learning script that picks up basic info from a doc file (about job title and location etc) 
and compiles a list of available job openings on Indeed, in an easy to use excel file. More details on the readme file.

There are a total of 4 files in this repository. One readme, two scripts and one doc file attached in this repository.

1 Classification_Model_comparison.py : This script allows me to collect some sample data and then compare the data on different machine learning algorithms. This would
					allow me to choose the algorithm with the highest accuracy towards the type of data being parsed.

2.Classification_Script.py: This is a kind of a superscript of the previous file with certain modifications in the data collection section of the code
			and a few additions towards the end. This code would use the algorithm of your choice, decided after running the previous script,
			to parse and store the job opening data, scraping page by page on Indeed website to get the data.

3. Cities1.doc : This file is the input file for both the above scripts.But first, Check either for 'cities1.doc' or 'cities.doc' in the code. Change that to 'Cities1.doc' 
		IMP: Use this file to cater to your specific job search. (Specific Job title and Locations you are targeting


If you need any more help, please reach out to me at vaibhavsingh1208@gmail.com


-------------------------Thank you------------------------------------------------------- 
