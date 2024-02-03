#pragma once

class IAlgorithm
{
public:
	void virtual runMaster() = 0;
	void virtual runWorker() = 0;
};
