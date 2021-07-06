#include "stdio.h"
#include "stdlib.h"

#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TVector3.h>

#include <RAT/DS/PMT.hh>
#include <RAT/DS/PMTInfo.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/Run.hh>
#include <RAT/DS/MCPMT.hh>

extern "C" {
double* retArr;
double num = 42.0;
double* square(double* arr, int size) {
  retArr = (double*)malloc(size * sizeof(double));
  for (int i = 0; i < size; i++) {
    retArr[i] = arr[i] * arr[i];
  }
  num += 1;
  return retArr;
}
void freeSquare() { free(retArr); }

int entries = 0;
int pmtcount = 0;
TFile* tfile;
TTree* runT;
TTree* T;
double* pmt_posx;
double* pmt_posy;
double* pmt_posz;

double* pmt_charge;
double* pmt_time;
int* mcpmt_noise;
int* pmt_type;

double boundary_radius;
double boundary_halfz;

RAT::DS::Run* run;
RAT::DS::PMTInfo* pmtinfo;
RAT::DS::Root* ds;

    
void calculateBoundaries()
{
  boundary_radius = 1.0;
  boundary_halfz = 1.0;
  for(int i=0; i<pmtcount; i++)
  {
    double x = pmt_posx[i];
    double y = pmt_posy[i];
    double z = pmt_posz[i];
    double rho = sqrt(x*x + y*y);
    int type = pmt_type[i];
    // Type 1 is inner, type 2 is veto
    if (type == 1)
    {
      if (rho > boundary_radius)
        boundary_radius = rho;
      if ( abs(z) > boundary_halfz )
        boundary_halfz = abs(z);
    }
  }
  // Apply a buffer
  double buffer = 10.0;
  boundary_radius += buffer;
  boundary_halfz += buffer;
}

void openFile(char* p) {
  // Default parameters
  tfile = new TFile(p,"UPDATE");
  // PMT Info
  runT = (TTree*)tfile->Get("runT");
  runT->SetBranchAddress("run", &run);
  runT->GetEntry(0);
  pmtinfo = run->GetPMTInfo();
  pmtcount = pmtinfo->GetPMTCount();
  printf("PMTs: %i\n", pmtcount);
  pmt_posx = (double*)malloc(pmtcount * sizeof(double));
  pmt_posy = (double*)malloc(pmtcount * sizeof(double));
  pmt_posz = (double*)malloc(pmtcount * sizeof(double));
  pmt_charge = (double*)malloc(pmtcount * sizeof(double));
  pmt_time = (double*)malloc(pmtcount * sizeof(double));
  pmt_type = (int*)malloc(pmtcount * sizeof(int));
  mcpmt_noise = (int*)malloc(pmtcount * sizeof(int));
for (int i=0; i < pmtcount; i++) {

}

  for (int i = 0; i < pmtcount; i++) {
    TVector3 v = pmtinfo->GetPosition(i);
    pmt_posx[i] = v.X();
    pmt_posy[i] = v.Y();
    pmt_posz[i] = v.Z();
    pmt_type[i] = pmtinfo->GetType(i);
  }

  T = (TTree*)tfile->Get("T");
  entries = T->GetEntries();
  printf("Entries: %i\n", entries);
  T->SetBranchAddress("ds", &ds);

  calculateBoundaries();
}

int getEntries() { return entries; }
double getBoundaryRadius() { return boundary_radius; }
double getBoundaryHalfz() { return boundary_halfz; }

RAT::DS::EV* ev;
int nhit = 0;
void getEvent(int event) {
  // clear last event;
  T->GetEvent(event);

  // reset pmt hits
  for (int i = 0; i < pmtcount; i++) {
    pmt_charge[i] = -100;
    pmt_time[i] = -100;
    mcpmt_noise[i] = -100;
  }
  // check first
  if (ds->GetEVCount() < 1) return;
  ev = ds->GetEV(0);
  nhit = ev->GetPMTCount();
  for (int i = 0; i < nhit; i++) {
    RAT::DS::PMT* apmt = ev->GetPMT(i);
    int pid = apmt->GetID();
    double charge = apmt->GetCharge();
    double time = apmt->GetTime();
    pmt_charge[pid] = charge;
    pmt_time[pid] = time;
  }
  RAT::DS::MC* mc = ds->GetMC();
  int mcpmt_count = mc->GetMCPMTCount();
  for (int i = 0; i < mcpmt_count; i++) {
    RAT::DS::MCPMT* mcpmt = mc->GetMCPMT(i);
    int mcpmt_id = mcpmt->GetID();
    int mcphoton_count = mcpmt->GetMCPhotonCount();
    if (mcphoton_count != 0) {RAT::DS::MCPhoton* mcphoton = mcpmt->GetMCPhoton(0);
        if (mcphoton->IsDarkHit()) mcpmt_noise[mcpmt_id] = 1;
        else mcpmt_noise[mcpmt_id] = 0;
    }
  }
}

std::vector<double> xtrack, ytrack, ztrack;
std::vector<int> pnames;
std::map<std::string, int> pnmap = {
    {"Cerenkov", 1}, {"Scintillation", 2}, {"Reemission", 2},  // from scint.
    {"e-", 3},       {"gamma", 4},         {"mu-", 5},        {"mu+", 5}};

void getTracking() {
  RAT::DS::MC* mc = ds->GetMC();
  int nTracks = mc->GetMCTrackCount();
  // Loop over each track
  xtrack.clear();
  ytrack.clear();
  ztrack.clear();
  pnames.clear();

  for (int trk = 0; trk < nTracks; trk++) {
    RAT::DS::MCTrack* track = mc->GetMCTrack(trk);
    std::string name = track->GetParticleName();
    if (name == "opticalphoton") name = track->GetMCTrackStep(0)->GetProcess();
    int nSteps = track->GetMCTrackStepCount();
    for (int stp = 0; stp < nSteps; stp++) {
      RAT::DS::MCTrackStep* step = track->GetMCTrackStep(stp);
      TVector3 tv = step->GetEndpoint();
      if ((stp != 0) && (stp != nSteps - 1)) {
        xtrack.push_back(tv.X());
        ytrack.push_back(tv.Y());
        ztrack.push_back(tv.Z());
        pnames.push_back(pnmap[name]);
      }
      xtrack.push_back(tv.X());
      ytrack.push_back(tv.Y());
      ztrack.push_back(tv.Z());
      pnames.push_back(pnmap[name]);
    }
  }
}

double* mc_x;
double* mc_y;
double* mc_z;
double* mc_u;
double* mc_v;
double* mc_w;
double* mc_ke;
double* mc_t;
int* mc_pdg;
int mcpcount;

void getMCTruth() {
  // Get MC indicent particle information
  RAT::DS::MC* mc = ds->GetMC();
  mcpcount = mc->GetMCParticleCount();
  mc_x = (double*)malloc(mcpcount * sizeof(double));
  mc_y = (double*)malloc(mcpcount * sizeof(double));
  mc_z = (double*)malloc(mcpcount * sizeof(double));
  mc_u = (double*)malloc(mcpcount * sizeof(double));
  mc_v = (double*)malloc(mcpcount * sizeof(double));
  mc_w = (double*)malloc(mcpcount * sizeof(double));
  mc_ke = (double*)malloc(mcpcount * sizeof(double));
  mc_t = (double*)malloc(mcpcount * sizeof(double));
  mc_pdg = (int*)malloc(mcpcount * sizeof(int));
  // Check if multiple particles were tracked
  for( int p=0; p<mcpcount; p++ ){  
    RAT::DS::MCParticle* particle = mc->GetMCParticle(p);
    TVector3 mcpos        = particle->GetPosition();
    mc_x[p] = mcpos.X();
    mc_y[p] = mcpos.Y();
    mc_z[p] = mcpos.Z();
    TVector3 mcmom        = particle->GetMomentum();
    mc_u[p] = mcmom.X();
    mc_v[p] = mcmom.Y();
    mc_w[p] = mcmom.Z();
    double kinetic_energy = particle->GetKE();
    mc_ke[p] = kinetic_energy;
    double event_time = particle->GetTime() - ev->GetDeltaT();
    mc_t[p] = event_time;
    int pdgcode           = particle->GetPDGCode();
    mc_pdg[p] = pdgcode;
  }
}

//mark PMT with id=pid in all events as broken by setting PMT Type=3
void maskPMT(int pid) {
  pmtinfo->SetType(pid,3);
  run->SetPMTInfo(pmtinfo);
  printf("PMT type after: %i\n",  pmtinfo->GetType(pid));
}
 
  // Write modified rat objects to a new file
void writeFile(char* p){
  TFile* newfile = new TFile(p, "recreate");
  TTree* newrunT = runT->CloneTree(0);
  TTree* newT = T->CloneTree();
  newrunT->Fill();
  newfile->Write();
}
    
    
double* getMCX() { return mc_x; }
double* getMCY() { return mc_y; }
double* getMCZ() { return mc_z; }
double* getMCU() { return mc_u; }
double* getMCV() { return mc_v; }
double* getMCW() { return mc_w; }
double* getMCKE() { return mc_ke; }
double* getMCT() { return mc_t; }
int* getPDG() { return mc_pdg; }
int getMCCount() { return mcpcount; }

int getTrackCount() { return xtrack.size(); }
double* getTrackX() { return &xtrack[0]; }
double* getTrackY() { return &ytrack[0]; }
double* getTrackZ() { return &ztrack[0]; }
int* getTrackNames() { return &pnames[0]; }

int getNHit() { return nhit; }
int getPMTCount() { return pmtcount; }
double* getPMTX() { return pmt_posx; }
double* getPMTY() { return pmt_posy; }
double* getPMTZ() { return pmt_posz; }
double* getCharge() { return pmt_charge; }
double* getTime() { return pmt_time; }
int* getPMTNoise() { return mcpmt_noise; }
int* getPMTType() { return pmt_type; }
}
