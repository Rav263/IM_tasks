#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
//#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/random-variable-stream.h"
using namespace ns3;


NS_LOG_COMPONENT_DEFINE("pro_ver_1");

uint32_t num_of_nodes = 3;
uint32_t packet_size = 1024;

void RandomFunction (void) {
    std::cout << "RandomFunction received event at "
             << Simulator::Now ().GetSeconds () << "s" << std::endl;
}



int main(int argc, char **argv) {
//    ExponentialVariable y(2902);
    
    CommandLine cmd;
    cmd.AddValue ("nCsma", "Number of \"extra\" CSMA nodes/devices", num_of_nodes);

    cmd.Parse (argc,argv);

    LogComponentEnable ("CsmaChannel", LOG_LEVEL_DEBUG);
    LogComponentEnable ("CsmaHelper", LOG_LEVEL_DEBUG);

    
    NodeContainer csma_nodes;
    csma_nodes.Create(num_of_nodes);
    
    CsmaHelper csma;
    csma.SetChannelAttribute ("DataRate", StringValue ("100Mbps"));
    csma.SetChannelAttribute ("Delay", TimeValue (NanoSeconds (6560)));

    NetDeviceContainer csma_devices;
    csma_devices = csma.Install (csma_nodes);

    InternetStackHelper stack;
    stack.Install(csma_nodes);

    Ipv4AddressHelper address;
    address.SetBase ("10.1.2.0", "255.255.255.0");
    Ipv4InterfaceContainer csmaInterfaces;
    csmaInterfaces = address.Assign (csma_devices);
  
    //UdpEchoServerHelper echoServer (9);

    /*ApplicationContainer serverApps = echoServer.Install (csma_nodes.Get (num_of_nodes - 1));
    serverApps.Start (Seconds (1.0));
    serverApps.Stop (Seconds (10.0));

    UdpEchoClientHelper echo_client (csmaInterfaces.GetAddress (num_of_nodes - 1), 9);
    echo_client.SetAttribute ("MaxPackets", UintegerValue (10));
    echo_client.SetAttribute ("PacketSize", UintegerValue (1024));
    echo_client.SetAttribute ("Interval", TimeValue (Seconds (1.0)));
*/
   
    OnOffHelper on_off_client("ns3::UdpSocketFactory", csmaInterfaces.GetAddress(num_of_nodes - 1));
    on_off_client.SetAttribute ("PacketSize", UintegerValue(packet_size));
    on_off_client.SetAttribute ("MaxBytes", UintegerValue (10 * packet_size));
    //on_off_client.SetAttribute("m_onTime", &RandomFunction);

    PacketSocketAddress socket;
    socket.SetSingleDevice (csma_devices.Get(0)->GetIfIndex());
    socket.SetPhysicalAddress (csma_devices.Get(1)->GetAddress());
    
    PacketSinkHelper sink = PacketSinkHelper ("ns3::PacketSocketFactory", socket);
    ApplicationContainer serverApps = sink.Install (csma_nodes.Get (num_of_nodes - 1));
    serverApps.Start (Seconds (0));
    serverApps.Stop (Seconds (10.0));

    ApplicationContainer clientApps = on_off_client.Install (csma_nodes.Get (0));
    for (uint32_t i = 1; i < num_of_nodes - 1; i++) {
        clientApps.Add(on_off_client.Install(csma_nodes.Get(i)));
    }
    
    clientApps.Start (Seconds (0.5));
    clientApps.Stop (Seconds (10.0));

    Ipv4GlobalRoutingHelper::PopulateRoutingTables ();


    csma.EnablePcap ("second", csma_devices.Get(0), true);

    Simulator::Run ();
    Simulator::Destroy ();
}
