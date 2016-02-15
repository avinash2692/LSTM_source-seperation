from numpy import *

# SDR, SIR, SAR estimation
def bss_eval( sep, i, sources):
	# Current target
	target = sources[i]
	
	# Target contribution
	s_target = target * dot( target, sep.T) / dot( target, target.T)
		
	# Interference contribution
	pse = dot( dot( sources, sep.T), \
						linalg.inv( dot( sources, sources.T))).T.dot( sources)
	e_interf = pse - s_target
	
	# Artifact contribution
	e_artif= sep - pse;
	
	# Interference + artifacts contribution
	e_total = e_interf + e_artif;
	
	# Computation of the log energy ratios
	sdr = 10*log10( sum( s_target**2) / sum( e_total**2));
	sir = 10*log10( sum( s_target**2) / sum( e_interf**2));
	sar = 10*log10( sum( (s_target + e_interf)**2) / sum( e_artif**2));
	
	# Done!
	return (sdr, sir, sar)


# and here’s the tset function:


