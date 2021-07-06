from sibyl_cpp import snake
import numpy as np
import sys
import pickle
import math
#To use this function just call the script and the number of the file at the end
i=sys.argv[1]
print(i)
#warning! doesnt check particle ID!
sourcefile=str(i)+".root"

min_pmt_hits=3
events=[]
labels=['x','y','z','zenith','azimuth','t','energy']
sn = snake()
event_range=[]

sn.openFile(sourcefile)
print('we made it')
for event in range(0,sn.getEntries()):
    sn.getEvent(event)
    charges, times ,noise = sn.getHitInfo()
    PMT_id=sn.getID()
    pmts=np.where(charges>-100)[0] #get index of nonzero values
    if (len(pmts) > 0 and len(np.where(PMT_id[pmts]==1)[0]) > min_pmt_hits):
        sn.getMCTruth()
        x,y,z=sn.getXYZ()
        p=np.column_stack(sn.getMCVal())
        zenith = math.acos(p[0,5]/np.linalg.norm(p[0,3:6],2))
        azimuth = math.atan2(p[0,4], p[0,3])+3.141592654
        p=np.column_stack((p[0,0],p[0,1],p[0,2],zenith,azimuth,p[0,6],p[0,7]))
        new_params = np.repeat(p, len(pmts), axis=0)
        new_hits = np.column_stack((x[pmts],y[pmts],z[pmts],times[pmts],charges[pmts],PMT_id[pmts],noise[pmts]))

        data={
            "hits":  new_hits,
            "total_charge": [np.sum(new_hits[:,4]),len(new_hits)],
            "truth": p[0]   
        }
        events.append(data)
    if(event%10000==0):
        print(sourcefile+": "+str(100*event/sn.getEntries())+"% complete.")
        print("-----------------------------------------------------------")

    
fileObj = open(str(i)+'.pkl', 'wb')
pickle.dump(events,fileObj)
fileObj.close()
