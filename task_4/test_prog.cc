#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/csma-module.h"


#include <iostream>



int main(int argc, char **argv) {
    ns3::Time::SetResolution (ns3::Time::NS);


    std::cout << "FUCK YOU NS3" << std::endl;
}
