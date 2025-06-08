import os
import gzip
import json
import time

from seleniumwire import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from io import BytesIO
from urllib.parse import urlparse
from itertools import product

# tickers = ['ACKB.BR', 'AED.BR', 'AGS.BR', 'ARGX.BR', 'AZE.BR', 'BEKB.BR', 'COFB.BR', 'COLR.BR', 'DIE.BR', 'ELI.BR', 'EURN.BR', 'FAGR.BR', 'GBLB.BR', 'KBCA.BR', 'MELE.BR', 'ONTEX.BR', 'PROX.BR', 'SHUR.BR', 'SOLB.BR', 'TNET.BR', 'UCB.BR', 'VGP.BR', 'WDP.BR', 'XIOR.BR', 'ALK-B.CO', 'ALMB.CO', 'AMBU-B.CO', 'MAERSK-B.CO', 'BAVA.CO', 'BIOPOR.CO', 'BOOZT-DKK.CO', 'AOJ-B.CO', 'CARL-B.CO', 'CBRAIN.CO', 'CHEMM.CO', 'COLO-B.CO', 'COLUM.CO', 'DNORD.CO', 'DANSKE.CO', 'DFDS.CO', 'FLS.CO', 'GMAB.CO', 'GN.CO', 'GREENH.CO', 'HLUN-A.CO', 'HLUN-B.CO', 'HH.CO', 'ISS.CO', 'JYSK.CO', 'MATAS.CO', 'NETC.CO', 'NKT.CO', 'NNIT.CO', 'NDA-DK.CO', 'NOVO-B.CO', 'ORSTED.CO', 'PNDORA.CO', 'RILBA.CO', 'ROCK-B.CO', 'RBREW.CO', 'RTX.CO', 'SHAPE.CO', 'SKJE.CO', 'SPNO.CO', 'SYDB.CO', 'TRMD-A.CO', 'TRYG.CO', 'VWS.CO', 'VJBA.CO', 'DEMANT.CO', 'ZEAL.CO', 'ANA.MC', 'ACX.MC', 'AENA.MC', 'AMP.MC', 'APPS.MC', 'AI.MC', 'A3M.MC', 'ADX.MC', 'BBVA.MC', 'SAN.MC', 'BKY.MC', 'BST.MC', 'CABK.MC', 'CLNX.MC', 'LOG.MC', 'CIE.MC', 'ANE.MC', 'EDR.MC', 'ENC.MC', 'GEST.MC', 'DOM.MC', 'GCO.MC', 'IBE.MC', 'ITX.MC', 'COL.MC', 'IAG.MC', 'ROVI.MC', 'MRL.MC', 'MTB.MC', 'NTGY.MC', 'NXT.MC', 'OHLA.MC', 'PHM.MC', 'SPH.MC', 'CASH.MC', 'SCYR.MC', 'TEF.MC', 'UBS.MC', 'VID.MC', 'BFF.MI', 'BGN.MI', 'IF.MI', 'BMPS.MI', 'BPE.MI', 'BPSO.MI', 'PRO.MI', 'BST.MI', 'BAMI.MI', 'BE.MI', 'BNP.MI', 'BC.MI', 'BZU.MI', 'CPR.MI', 'CRL.MI', 'CEM.MI', 'CNHI.MI', 'CE.MI', 'DIS.MI', 'DAN.MI', 'DAL.MI', 'DLG.MI', 'DEA.MI', 'DBK.MI', 'DOV.MI', 'ELN.MI', 'ENAV.MI', 'ENI.MI', 'EPR.MI', 'EXAI.MI', 'FNX.MI', 'RACE.MI', 'FILA.MI', 'FCT.MI', 'FBK.MI', 'US.MI', 'FCM.MI', 'FUL.MI', 'GAMB.MI', 'GE.MI', 'GEO.MI', 'GO.MI', 'GVS.MI', 'HER.MI', 'IE.MI', 'ILTY.MI', 'IGD.MI', 'DNR.MI', 'IP.MI', 'IRE.MI', 'ITW.MI', 'IIG.MI', 'IVG.MI', 'JUVE.MI', 'LR.MI', 'LDO.MI', 'MT.MI', 'MARR.MI', 'MB.MI', 'MFEA.MI', 'MFEB.MI', 'MONC.MI', 'NSP.MI', 'NWL.MI', 'OJM.MI', 'PRL.MI', 'PIRC.MI', 'PRM.MI', 'PRY.MI', 'RWAY.MI', 'RST.MI', 'SFL.MI', 'SPM.MI', 'SCF.MI', 'SFER.MI', 'SL.MI', 'IOT.MI', 'SERI.MI', 'SRG.MI', 'SO.MI', 'STLAM.MI', 'STM.MI', 'TGYM.MI', 'TPRO.MI', 'TIT.MI', 'TITR.MI', 'TEF.MI', 'TEN.MI', 'TRN.MI', 'TKA.MI', 'TNXT.MI', 'TOD.MI', 'TXT.MI', 'UCG.MI', 'DAPP.MI', 'VNT.MI', 'WBD.MI', 'ZV.MI', 'A2A.MI', 'AED.MI', 'ALA.MI', 'AMZN.MI', 'AMP.MI', 'ANIM.MI', 'AV.MI', 'ARIS.MI', 'ASC.MI', 'G.MI', 'AGL.MI', 'AVIO.MI', '2020.OL', 'ABG.OL', 'AFG.OL', 'AMSC.OL', 'ABT.OL', 'ARCH.OL', 'ARR.OL', 'ASA.OL', 'BCS.OL', 'BORR.OL', 'DNB.OL', 'ENTRA.OL', 'EQNR.OL', 'EPR.OL', 'FLNG.OL', 'FRO.OL', 'GOGL.OL', 'GSF.OL', 'HEX.OL', 'JIN.OL', 'NAPA.OL', 'NEXT.OL', 'NAS.OL', 'NRC.OL', 'PEN.OL', 'PHO.OL', 'SAGA.OL', 'SATS.OL', 'SDRL.OL', 'SBO.OL', 'SPOL.OL', 'STB.OL', 'STRO.OL', 'TEL.OL', 'TGS.OL', 'VOW.OL', 'VGM.OL', 'WEST.OL', 'ZAL.OL', 'AALB.AS', 'ABN.AS', 'AXS.AS', 'ADYEN.AS', 'AD.AS', 'AKZA.AS', 'ALFEN.AS', 'ALLFG.AS', 'AMG.AS', 'APAM.AS', 'ARCAD.AS', 'MT.AS', 'ASM.AS', 'ASML.AS', 'ASRNL.AS', 'AVTX.AS', 'BAMNB.AS', 'BFIT.AS', 'CCEP.AS', 'CTPNV.AS', 'DSFIR.AS', 'EXO.AS', 'FAST.AS', 'FUR.AS', 'GLPG.AS', 'HEIJM.AS', 'HEIA.AS', 'HEIO.AS', 'IMCD.AS', 'INGA.AS', 'INPST.AS', 'JDEP.AS', 'TKWY.AS', 'KPN.AS', 'MAREL.AS', 'NN.AS', 'PSH.AS', 'PHARM.AS', 'PHIA.AS', 'PNL.AS', 'PRX.AS', 'REN.AS', 'RWI.AS', 'SBMO.AS', 'SHELL.AS', 'LIGHT.AS', 'TWEKA.AS', 'TOM2.AS', 'URW.AS', 'UNA.AS', 'UMG.AS', 'VLK.AS', 'VPK.AS', 'WHA.AS', 'WKL.AS', 'AAK.ST', 'ABB.ST', 'ACAD.ST', 'ATIC.ST', 'ALIF-B.ST', 'ANOD-B.ST', 'ADDT-B.ST', 'ALFA.ST', 'ALIG.ST', 'ATORX.ST', 'AMBEA.ST', 'AQ.ST', 'ARISE.ST', 'ARJO-B.ST', 'ASSA-B.ST', 'AZN.ST', 'ATCO-A.ST', 'ATCO-B.ST', 'ATRLJ-B.ST', 'ATT.ST', 'AXFO.ST', 'B3.ST', 'BALCO.ST', 'BEGR.ST', 'BEIA-B.ST', 'BEIJ-B.ST', 'BETS-B.ST', 'BETCO.ST', 'BILI-A.ST', 'BILL.ST', 'BIOA-B.ST', 'BIOG-B.ST', 'BONAV-B.ST', 'BONEX.ST', 'BOOZT.ST', 'BUFAB.ST', 'BULTEN.ST', 'BURE.ST', 'BHG.ST', 'CRAD-B.ST', 'CALTX.ST', 'CANTA.ST', 'CAST.ST', 'CATE.ST', 'CLAS-B.ST', 'CLA-B.ST', 'COLL.ST', 'COIC.ST', 'COOR.ST', 'CORE-PREF.ST', 'CRED-A.ST', 'DEDI.ST', 'DIOS.ST', 'DOM.ST', 'DORO.ST', 'DUNI.ST', 'ELUX-B.ST', 'EPRO-B.ST', 'EKTA-B.ST', 'ENQ.ST', 'EOLU-B.ST', 'EPI-A.ST', 'EPI-B.ST', 'EQT.ST', 'ESSITY-B.ST', 'EVO.ST', 'FABG.ST', 'BALD-B.ST', 'FPAR-A.ST', 'FING-B.ST', 'G5EN.ST', 'GIGSEK.ST', 'GETI-B.ST', 'GRNG.ST', 'HNSA.ST', 'HANZA.ST', 'HM-B.ST', 'HEXA-B.ST', 'HTRO.ST', 'HPOL-B.ST', 'HMS.ST', 'HOFI.ST', 'HOLM-B.ST', 'HUFV-A.ST', 'HUM.ST', 'HUSQ-B.ST', 'IAR-B.ST', 'IMMNOV.ST', 'INDU-C.ST', 'INDU-A.ST', 'INSTAL.ST', 'IPCO.ST', 'INTRUM.ST', 'LATO-B.ST', 'INVE-A.ST', 'INVE-B.ST', 'IVSO.ST', 'INWI.ST', 'JM.ST', 'KAR.ST', 'KIND-SDB.ST', 'KINV-B.ST', 'LAGR-B.ST', 'LIFCO-B.ST', 'LIAB.ST', 'LOOMIS.ST', 'LUND-B.ST', 'LUG.ST', 'LUMI.ST', 'MCOV-B.ST', 'MEKO.ST', 'TIGO-SDB.ST', 'MIPS.ST', 'MTG-B.ST', 'MMGR-B.ST', 'MTRS.ST', 'MYCR.ST', 'NCC-B.ST', 'NETI-B.ST', 'NEWA-B.ST', 'NGS.ST', 'NIBE-B.ST', 'NOBI.ST', 'NOLA-B.ST', 'NDA-SE.ST', 'NP3.ST', 'OEM-B.ST', 'PNDX-B.ST', 'PEAB-B.ST', 'PFE.ST', 'PLAZ-B.ST', 'PREV-B.ST', 'PRIC-B.ST', 'QLINEA.ST', 'RATO-B.ST', 'RAY-B.ST', 'REJL-B.ST', 'RESURS.ST', 'SAAB-B.ST', 'SAGA-B.ST', 'SAGA-D.ST', 'SBB-B.ST', 'SAND.ST', 'SCST.ST', 'SHOT.ST', 'SECT-B.ST', 'SECU-B.ST', 'SINT.ST', 'SEB-C.ST', 'SEB-A.ST', 'SKA-B.ST', 'SKF-B.ST', 'SKIS-B.ST', 'SSAB-B.ST', 'SSAB-A.ST', 'STAR-B.ST', 'STE-R.ST', 'SCA-B.ST', 'SHB-A.ST', 'SHB-B.ST', 'SVOL-B.ST', 'SWEC-B.ST', 'SWED-A.ST', 'SOBI.ST', 'SYSR.ST', 'TEL2-B.ST', 'ERIC-B.ST', 'TELIA.ST', 'TETY.ST', 'THULE.ST', 'TOBII.ST', 'TREL-B.ST', 'TROAX.ST', 'VIT-B.ST', 'VITR.ST', 'VOLO.ST', 'VOLV-A.ST', 'VOLV-B.ST', 'VNV.ST', 'WALL-B.ST', 'WIHL.ST', 'XBRANE.ST', 'XVIVO.ST', 'AB.PA', 'ADP.PA', 'AI.PA', 'AIR.PA', 'ALD.PA', 'ALO.PA', 'AMUN.PA', 'APAM.PA', 'MT.PA', 'AKE.PA', 'ATO.PA', 'AUB.PA', 'AVT.PA', 'CS.PA', 'BEN.PA', 'BB.PA', 'BIG.PA', 'BNP.PA', 'BVI.PA', 'CAP.PA', 'CO.PA', 'CRI.PA', 'CLARI.PA', 'COFA.PA', 'ACA.PA', 'BN.PA', 'AM.PA', 'EDEN.PA', 'ELIOR.PA', 'ELIS.PA', 'ENGI.PA', 'ALESK.PA', 'EL.PA', 'ES.PA', 'RF.PA', 'ERF.PA', 'FRVIA.PA', 'GTT.PA', 'RMS.PA', 'NK.PA', 'IPH.PA', 'IPN.PA', 'IPS.PA', 'DEC.PA', 'KOF.PA', 'KER.PA', 'LI.PA', 'OR.PA', 'FDJ.PA', 'LR.PA', 'MC.PA', 'MEDCL.PA', 'MERY.PA', 'ML.PA', 'NANO.PA', 'NEOEN.PA', 'NXI.PA', 'NRG.PA', 'ORA.PA', 'OSE.PA', 'OVH.PA', 'VAC.PA', 'POM.PA', 'PUB.PA', 'RCO.PA', 'RNO.PA', 'SAN.PA', 'SLB.PA', 'SU.PA', 'SCR.PA', 'SESG.PA', 'GLE.PA', 'SW.PA', 'SOI.PA', 'S30.PA', 'SPIE.PA', 'STLAP.PA', 'STMPA.PA', 'TE.PA', 'TEP.PA', 'HO.PA', 'TKO.PA', 'TTE.PA', 'TRI.PA', 'URW.PA', 'FR.PA', 'VK.PA', 'VLA.PA', 'VIE.PA', 'VRLA.PA', 'DG.PA', 'VIV.PA', 'VLTSA.PA', 'MF.PA', 'WLN.PA', 'XFAB.PA']
tickers_brussels = ['ENXTBR:ACKB', 'ENXTBR:AED', 'ENXTBR:AGS', 'ENXTBR:ARGX', 'ENXTBR:AZE', 'ENXTBR:BEKB', 
 'ENXTBR:COFB', 'ENXTBR:COLR', 'ENXTBR:DIE', 'ENXTBR:ELI', 'ENXTBR:EURN', 'ENXTBR:FAGR', 
 'ENXTBR:GBLB', 'ENXTBR:KBCA', 'ENXTBR:MELE', 'ENXTBR:ONTEX', 'ENXTBR:PROX', 'ENXTBR:SHUR', 
 'ENXTBR:SOLB', 'ENXTBR:TNET', 'ENXTBR:UCB', 'ENXTBR:VGP', 'ENXTBR:WDP', 'ENXTBR:XIOR']
tickers_madrid = ['BME:ANA', 'BME:ACX', 'BME:AENA', 'BME:AMP', 'BME:APPS', 'BME:AI', 'BME:A3M', 'BME:ADX',
 'BME:BBVA', 'BME:SAN', 'BME:BKY', 'BME:BST', 'BME:CABK', 'BME:CLNX', 'BME:LOG', 'BME:CIE',
 'BME:ANE', 'BME:EDR', 'BME:ENC', 'BME:GEST', 'BME:DOM', 'BME:GCO', 'BME:IBE', 'BME:ITX',
 'BME:COL', 'BME:IAG', 'BME:ROVI', 'BME:MRL', 'BME:MTB', 'BME:NTGY', 'BME:NXT', 'BME:OHLA',
 'BME:PHM', 'BME:SPH', 'BME:CASH', 'BME:SCYR', 'BME:TEF', 'BME:UBS', 'BME:VID']
tickers_milan = ['BIT:BFF', 'BIT:BGN', 'BIT:IF', 'BIT:BMPS', 'BIT:BPE', 'BIT:BPSO', 'BIT:PRO', 'BIT:BST',
 'BIT:BAMI', 'BIT:BE', 'BIT:BNP', 'BIT:BC', 'BIT:BZU', 'BIT:CPR', 'BIT:CRL', 'BIT:CEM',
 'BIT:CNHI', 'BIT:CE', 'BIT:DIS', 'BIT:DAN', 'BIT:DAL', 'BIT:DLG', 'BIT:DEA', 'BIT:DBK',
 'BIT:DOV', 'BIT:ELN', 'BIT:ENAV', 'BIT:ENI', 'BIT:EPR', 'BIT:EXAI', 'BIT:FNX', 'BIT:RACE',
 'BIT:FILA', 'BIT:FCT', 'BIT:FBK', 'BIT:US', 'BIT:FCM', 'BIT:FUL', 'BIT:GAMB', 'BIT:GE',
 'BIT:GEO', 'BIT:GO', 'BIT:GVS', 'BIT:HER', 'BIT:IE', 'BIT:ILTY', 'BIT:IGD', 'BIT:DNR',
 'BIT:IP', 'BIT:IRE', 'BIT:ITW', 'BIT:IIG', 'BIT:IVG', 'BIT:JUVE', 'BIT:LR', 'BIT:LDO',
 'BIT:MT', 'BIT:MARR', 'BIT:MB', 'BIT:MFEA', 'BIT:MFEB', 'BIT:MONC', 'BIT:NSP', 'BIT:NWL',
 'BIT:OJM', 'BIT:PRL', 'BIT:PIRC', 'BIT:PRM', 'BIT:PRY', 'BIT:RWAY', 'BIT:RST', 'BIT:SFL',
 'BIT:SPM', 'BIT:SCF', 'BIT:SFER', 'BIT:SL', 'BIT:IOT', 'BIT:SERI', 'BIT:SRG', 'BIT:SO',
 'BIT:STLAM', 'BIT:STM', 'BIT:TGYM', 'BIT:TPRO', 'BIT:TIT', 'BIT:TITR', 'BIT:TEF',
 'BIT:TEN', 'BIT:TRN', 'BIT:TKA', 'BIT:TNXT', 'BIT:TOD', 'BIT:TXT', 'BIT:UCG', 'BIT:DAPP',
 'BIT:VNT', 'BIT:WBD', 'BIT:ZV', 'BIT:A2A', 'BIT:AED', 'BIT:ALA', 'BIT:AMZN', 'BIT:AMP',
 'BIT:ANIM', 'BIT:AV', 'BIT:ARIS', 'BIT:ASC', 'BIT:G', 'BIT:AGL', 'BIT:AVIO']
tickers_oslo = ['OB:ABG', 'OB:AFG', 'OB:AMSC', 'OB:ABT', 'OB:ARCH', 'OB:ARR', 'OB:ASA', 'OB:BCS', 'OB:BORR',
 'OB:DNB', 'OB:ENTRA', 'OB:EQNR', 'OB:EPR', 'OB:FLNG', 'OB:FRO', 'OB:GOGL', 'OB:GSF',
 'OB:HEX', 'OB:JIN', 'OB:NAPA', 'OB:NEXT', 'OB:NAS', 'OB:NRC', 'OB:PEN', 'OB:PHO',
 'OB:SAGA', 'OB:SATS', 'OB:SDRL', 'OB:SBO', 'OB:SPOL', 'OB:STB', 'OB:STRO', 'OB:TEL',
 'OB:TGS', 'OB:VOW', 'OB:VGM', 'OB:WEST', 'OB:ZAL']
tickers_amsterdam = ['ENXTAM:AALB', 'ENXTAM:ABN', 'ENXTAM:AXS', 'ENXTAM:ADYEN', 'ENXTAM:AD', 'ENXTAM:AKZA',
 'ENXTAM:ALFEN', 'ENXTAM:ALLFG', 'ENXTAM:AMG', 'ENXTAM:APAM', 'ENXTAM:ARCAD', 'ENXTAM:MT',
 'ENXTAM:ASM', 'ENXTAM:ASML', 'ENXTAM:ASRNL', 'ENXTAM:AVTX', 'ENXTAM:BAMNB', 'ENXTAM:BFIT',
 'ENXTAM:CCEP', 'ENXTAM:CTPNV', 'ENXTAM:DSFIR', 'ENXTAM:EXO', 'ENXTAM:FAST', 'ENXTAM:FUR',
 'ENXTAM:GLPG', 'ENXTAM:HEIJM', 'ENXTAM:HEIA', 'ENXTAM:HEIO', 'ENXTAM:IMCD', 'ENXTAM:INGA',
 'ENXTAM:INPST', 'ENXTAM:JDEP', 'ENXTAM:TKWY', 'ENXTAM:KPN', 'ENXTAM:MAREL', 'ENXTAM:NN',
 'ENXTAM:PSH', 'ENXTAM:PHARM', 'ENXTAM:PHIA', 'ENXTAM:PNL', 'ENXTAM:PRX', 'ENXTAM:REN',
 'ENXTAM:RWI', 'ENXTAM:SBMO', 'ENXTAM:SHELL', 'ENXTAM:LIGHT', 'ENXTAM:TWEKA', 'ENXTAM:TOM2',
 'ENXTAM:URW', 'ENXTAM:UNA', 'ENXTAM:UMG', 'ENXTAM:VLK', 'ENXTAM:VPK', 'ENXTAM:WHA',
 'ENXTAM:WKL']
tickers_paris = ['ENXTPA:AB', 'ENXTPA:ADP', 'ENXTPA:AI', 'ENXTPA:AIR', 'ENXTPA:ALD', 'ENXTPA:ALO', 'ENXTPA:AMUN',
 'ENXTPA:APAM', 'ENXTPA:MT', 'ENXTPA:AKE', 'ENXTPA:ATO', 'ENXTPA:AUB', 'ENXTPA:AVT', 'ENXTPA:CS',
 'ENXTPA:BEN', 'ENXTPA:BB', 'ENXTPA:BIG', 'ENXTPA:BNP', 'ENXTPA:BVI', 'ENXTPA:CAP', 'ENXTPA:CO',
 'ENXTPA:CRI', 'ENXTPA:CLARI', 'ENXTPA:COFA', 'ENXTPA:ACA', 'ENXTPA:BN', 'ENXTPA:AM',
 'ENXTPA:EDEN', 'ENXTPA:ELIOR', 'ENXTPA:ELIS', 'ENXTPA:ENGI', 'ENXTPA:ALESK', 'ENXTPA:EL',
 'ENXTPA:ES', 'ENXTPA:RF', 'ENXTPA:ERF', 'ENXTPA:FRVIA', 'ENXTPA:GTT', 'ENXTPA:RMS', 'ENXTPA:NK',
 'ENXTPA:IPH', 'ENXTPA:IPN', 'ENXTPA:IPS', 'ENXTPA:DEC', 'ENXTPA:KOF', 'ENXTPA:KER', 'ENXTPA:LI',
 'ENXTPA:OR', 'ENXTPA:FDJ', 'ENXTPA:LR', 'ENXTPA:MC', 'ENXTPA:MEDCL', 'ENXTPA:MERY', 'ENXTPA:ML',
 'ENXTPA:NANO', 'ENXTPA:NEOEN', 'ENXTPA:NXI', 'ENXTPA:NRG', 'ENXTPA:ORA', 'ENXTPA:OSE',
 'ENXTPA:OVH', 'ENXTPA:VAC', 'ENXTPA:POM', 'ENXTPA:PUB', 'ENXTPA:RCO', 'ENXTPA:RNO',
 'ENXTPA:SAN', 'ENXTPA:SLB', 'ENXTPA:SU', 'ENXTPA:SCR', 'ENXTPA:SESG', 'ENXTPA:GLE', 'ENXTPA:SW',
 'ENXTPA:SOI', 'ENXTPA:S30', 'ENXTPA:SPIE', 'ENXTPA:STLAP', 'ENXTPA:STMPA', 'ENXTPA:TE',
 'ENXTPA:TEP', 'ENXTPA:HO', 'ENXTPA:TKO', 'ENXTPA:TTE', 'ENXTPA:TRI', 'ENXTPA:URW', 'ENXTPA:FR',
 'ENXTPA:VK', 'ENXTPA:VLA', 'ENXTPA:VIE', 'ENXTPA:VRLA', 'ENXTPA:DG', 'ENXTPA:VIV',
 'ENXTPA:VLTSA', 'ENXTPA:MF', 'ENXTPA:WLN', 'ENXTPA:XFAB']
tickers = tickers_brussels + tickers_madrid + tickers_milan + tickers_oslo + tickers_amsterdam + tickers_paris
# tickers.reverse()

fundamentals = [
    "pe_ltm", "current_ratio", "roa", "roe", "total_debt", "total_rev", "ev_to_ebitda_ltm", 
    "fcf_yield_ltm", "marketcap", "price_to_book"
]

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

def get_filepath(filename):
    directory = os.path.join("outputs", "fundamentals")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, filename)

def save_fundamental_response(filepath, filename, parsed_json):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(parsed_json, f, indent=2)
        print(f"{filename}: Saved ✅")

for fundamental, ticker in product(fundamentals, tickers):
    print(f"Processing {fundamental} for {ticker}...")

    filename = f"{fundamental}_{ticker}.json"
    filepath = get_filepath(filename)
    
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping...")
        continue

    url = f"https://finbox.com/{ticker}/explorer/{fundamental}"

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    try:
        # Wait for the dropdown and click it
        wait = WebDriverWait(driver, 20)
        dropdown = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Fiscal Years')]")))
        dropdown.click()

        # Wait for the full dropdown to render
        time.sleep(1)

        # Use an XPath that EXACTLY matches "Fiscal Quarters"
        fq_xpath = "//div[@role='button' and normalize-space(text())='Fiscal Quarters']"
        fq_button = wait.until(EC.presence_of_element_located((By.XPATH, fq_xpath)))

        # Click on 'Fiscal Quarters'. Use JavaScript to force-click in case it's not interactable normally
        driver.execute_script("arguments[0].click();", fq_button)

        # Give it time to trigger the network requests
        time.sleep(3)

        # Look for the correct POST request with timeseries and pe_ltm
        for request in reversed(driver.requests):
            if (
                request.method == 'POST' and
                "/_/api/v5/query" in request.url and
                request.response and
                'application/json' in request.response.headers.get('Content-Type', '')
            ):
                try:
                    # Decode gzipped body if needed
                    body = request.response.body
                    if request.response.headers.get('Content-Encoding') == 'gzip':
                        with gzip.GzipFile(fileobj=BytesIO(body)) as gz:
                            content = gz.read().decode('utf-8')
                    else:
                        content = body.decode('utf-8')

                    parsed_json = json.loads(content)

                    chart_data = parsed_json.get("data", {}).get("company", {}).get("glossary", {}).get("chart", {})
                    if chart_data.get("type") == "timeseries" and fundamental in chart_data.get("metrics", []):
                        save_fundamental_response(filepath, filename, parsed_json)
                        break
                except Exception as e:
                    print(f"Error parsing request: {e}")
        else:
            print(f"❌ No matching {fundamental} timeseries request found.")
    
    finally:
        driver.quit()