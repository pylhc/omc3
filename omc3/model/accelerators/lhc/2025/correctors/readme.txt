-------------- From Stephane ---------------------------------------------------

IR2/8
For IR2/8 optics re-matching *at injection*  we generally exclude both Q4 and Q5 on the injected beam side and the triplet (i.e. not only kq5.l2b1 and kq5.r8b2), 
in order to preserve the twiss parameter at the injection point (TL matching), the MKI/TDI vertical phase advance, and the TL trajectory till the injection point (Q5 matters).

In your context (optics correction), I agree that it maybe counter-productive to exclude the IT and Q4 
(of course putting some good weight for the BPM data near the injection point and the TDI), only the Q5 circuits makes sens

Therefore I do not also see any reason why excluding these Q5 circuits in the ramp, FT, collision, etc..


IR6
I would definitely exclude kq4.l6b1 and kq4.r6b2 because they sensibly (~10%) participate to the extraction kick 
(V-defocusing quadrupoles, without which the MKD would not be strong enough at 7 TeV). 
So to be excluded a priori so all the time (injection, ramp, FT, etc..).

But Indeed I do not see any reason why excluding kq4.l6b2. To be checked with Chiara, maybe?


IR1/5
Q4 in IR5 should be excluded by construction (RP-Q4off optics). 
For squeezed optics (beta-presqueezed=60 cm or below, so as of EoS@1.2 m), I would exclude Q6 both in IR1/5 (very small gradient) 
and Q5 in IR5 (very small gradient). 
For OMC knobs below 1.2 m, 
I would then try to work at cst Q4/5/6 in IR1&IR5 (in order to preserve the telescopic features of these optics). 
If you prefer, ideally the IT/Q4/5/6 setting correction should be optimized and kept constant, based on the 60 cm pivot optics (non telescopic)… 
but in practice I do not know wether this would not be an overhead for the usual steps which are taken.
----------------------------------------------------------------------------------------------------

So in short for squeezed optics:
 - To keep ATS features intact: q4, q5, q6 off in IP1 and IP5 
 - q4 in IP5 off by optics 
 - Best to keep off always: q6 in IP1 and IP5, q5 in IP5  (-> not in the MQT_TOP)
 - Best to keep off if possible: q4 and q5 in IP1 (-> not in the MQT_TOP)
 - kq4.l6b1 and kq4.r6b2 -> Do not use, they are between dump kicker and septum

For injection optics:
 - q4 in IP5 off by optics
 - kq4.l6b1 and kq4.r6b2 -> Do not use, they are between dump kicker and septum
 - kq5.l2b1 and kq5.r8b2 -> Do not use at injection, they are within the injection bump

Changes 2025 to 2024:
 - kq4.l6b2 -> now used, not sure why excluded in the past (copy/paste error)
 - kqt13.l7b1 -> now used, no idea why excluded in the past (!! Brought back in the main json. Cleaner to have it there !!)
 - Inj: 
   Removed kq4 in IP5, added kq4 in IP1 due to polarity switch.
   In the optics these magnets are turned off.
 - Top:
   Removed kq6 in IP1, as we are not supposed to use them. (see email above) 
