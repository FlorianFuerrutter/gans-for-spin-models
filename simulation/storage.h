#pragma once
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream> 
#include <string>

#include "defines.h"

//data seperator space
std::vector<std::vector<PRECISION>> loadFloatData(const std::string filename, bool skip_header = false)
{
	std::vector<std::vector<PRECISION>> data;

	//open file
	std::ifstream file(DATA_PATH + filename);
	if (!file.is_open())
	{
		std::cout << "[loadFloatData]: file not opened: " + std::string(DATA_PATH) + filename << std::endl;
		return data;
	}

	std::string str;
	if (skip_header)
		std::getline(file, str);
	
	while (std::getline(file, str))
	{
		std::stringstream ss;
		ss << str;

		//example data_line: t x1 y1 x2 y2 x3 y3
		std::vector<PRECISION> data_line;

		std::string temp;
		while (!ss.eof())
		{
			ss >> temp;

			double found = 0;
			if (std::stringstream(temp) >> found)
				data_line.push_back(found);
				//std::cout << found << " ";

			temp = "";
		}
		data.push_back(data_line);
	}

	return data;
}

//data seperator space
void storeFloatData(const std::vector<std::vector<PRECISION>>& data, const std::string filename, const std::string header = "")
{
	//open file
	std::ofstream file(DATA_PATH + filename);
	if (!file.is_open())
	{
		std::cout << "[storeFloatData]: file not opened: " + std::string(DATA_PATH) + filename << std::endl;
		return;
	}

	if (!header.empty())
		file << header << std::endl;

	for (int i = 0; i < data.size(); i++)
	{
		for (int j = 0; j < data.at(i).size(); j++)
			file << data.at(i).at(j) << " ";

		file << std::endl;
	}
}

//data seperator space
void storeStateData(const std::vector<std::array<int8_t, N>>& data, const std::string filename, const std::string header = "")
{
	//open file
	std::ofstream file(DATA_PATH + filename);
	if (!file.is_open())
	{
		std::cout << "[storeStateData]: file not opened: " + std::string(DATA_PATH) + filename << std::endl;
		return;
	}

	if (!header.empty())
		file << header << std::endl;

	for (int i = 0; i < data.size(); i++)
	{
		for (int j = 0; j < data.at(i).size(); j++)
			file << int(data.at(i).at(j)) << " ";

		file << std::endl;
	}
}