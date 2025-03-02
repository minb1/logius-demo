# Digikoppeling-koppelvlakstandaarden en voorschriften

## Overzicht

De Digikoppeling Architectuur legde in de eerdere versies grote nadruk op bevragingen en meldingen en legde een verband tussen deze interactiepatronen en de onderliggende standaarden, ('WUS voor bevragingen, ebMS voor meldingen en kennisgevingen'). Dit verband bleek in de praktijk niet altijd werkbaar of wenselijk. In 2020 is daarom besloten om de richtlijnen voor het toepassen van de Digikoppeling standaarden te wijzigen.

![Overzicht Digikoppeling Koppelvlakken](media/fig-structuurv2.png "Overzicht Digikoppeling Koppelvlakken")

Digikoppeling kent vier koppelvlakstandaarden

- WUS voor synchrone uitwisseling van gestructureerde berichten;

- ebMS2 voor asynchrone uitwisseling voor betrouwbaar berichtenverkeer;

- REST API voor synchrone gegevensuitwisseling met resources;

- Grote berichten voor het uitwisselen van grote bestanden;

De Digikoppeling-koppelvlakstandaarden beschrijven verschillende profielen. Elk profiel biedt een combinatie van kenmerken die in een bepaalde functionele behoefte voorziet.

De volgende profielen zijn onderkend:

- Best effort – geschikt voor bevragingen 

- Betrouwbaar (reliable) – geschikt voor meldingen 


Deze komen in de volgende varianten voor:

- Standaard (niets) – best effort of reliable

- Signed – geschikt voor de ondertekening van berichten

- Encrypted – geschikt voor de versleuteling van de payload en attachments (bericht-niveau security)

Door het gebruik van deze profielen worden deze aspecten correct afgehandeld en kunnen partijen sneller een koppelvlakstandaard implementeren.


| Onderdeel | Toelichting|
|---|---|
| Koppelvlakstandaard WUS | het gebruik van WUS voor synchrone uitwisseling van gestructureerde berichten en de WUS profielen.|
| Koppelvlakstandaard ebMS2 | Het gebruik van ebMS2 voor asynchrone uitwisseling en de ebMS2 profielen|
|Koppelvlakstandaard REST API| Het gebruik van REST APIs voor het synchroon raadplegen en bewerken van resources|
| Koppelvlakstandaard Grote Berichten | De uitwisseling van grote berichten maakt gebruik van WUS, ebMS2 of (indien gewenst) REST met HTTPS bestandsoverdracht |
| Beveiligingstandaarden en voorschriften  | Beschrijft de beveiligingstandaarden (TLS, signing en encryption) voor de Digikoppeling profielen WUS, ebMS2 en Grote berichten |
| Identificatie en Authenticatie | Beschrijft de identificatie van partijen, het opzetten van een tweezijdige beveiligde TLS-verbinding en over het ondertekenen  en versleutelen van berichten en bijlagen. |
| Overzicht Actuele Documentatie en Compliance | Overzicht van de actuele versie van de  Digikoppeling specificaties (normatief en niet-normatief)  |
| Gebruik en Achtergrond Digikoppeling Certificaten | Beschrijft de werking en gebruik van PKIoverheid Certificaten (niet-normatief) |


Tabel 6.1: Digikoppeling-standaarden

## Digikoppeling-voorschriften

Enkele afspraken over de functionaliteit van Digikoppeling hebben betrekking op de Digikoppeling-keten als geheel waar behalve de koppelvlakstandaarden ook partijen, intermediairs e.d. een onderdeel van vormen. En voor elke keten geldt dat deze ‘zo sterk is als de zwakste schakel’.

Onderstaande voorschriften gelden voor de hele Digikoppeling-keten. Partijen moeten er in hun eigen organisatie voor zorgen dat hun systemen, applicaties en toegang voor gebruikers aan de eisen voldoen.



| Aspect | Voorschrift | Toepassing en uitleg |
|---|---|---|
| Identiteit, authenticatie en autorisatie | Identificatie en authenticatie van partijen (ook intermediairs) vindt plaats in overeenstemming met het beleid hiervoor. Zowel service aanbieder als service afnemer moeten overeenkomstig afspraken autoriseren. De autorisatie gebeurt op organisatieniveau, niet op medewerkerniveau. | Beleid staat uitgewerkt in het document “Digikoppeling Identificatie en Authenticatie”. Een praktische werkwijze is uitgewerkt in het document “Gebruik en achtergrond Digikoppeling certificaten”. Autoriseren kan afhankelijk van noodzaak tweezijdig afgesproken worden. Immers bijvoorbeeld ook het stellen van een vraag kan al vertrouwelijk zijn.                                                                                                                             |
| Betrouwbaarheid en beschikbaarheid (reliability)  | Alle componenten in de Digikoppeling-keten dienen de betrouwbaarheid en beschikbaarheid van het berichtenverkeer in de keten te handhaven, met name door het gebruik van een betrouwbaar profiel. Het gaat hier specifiek om de betrouwbare aflevering van berichten via reliable messaging (het gaat dus niet om de beschikbaarheid of betrouwbaarheid van de applicaties in de keten). | Een betrouwbaar profiel garandeert dat een bericht met zekerheid (precies één keer) wordt afgeleverd en dat berichten zo mogelijk in de juiste volgorde worden afgeleverd, ook als de ontvanger tijdelijk niet beschikbaar is. Tussenliggende intermediairs maar ook de Digikoppeling-adapters bij de partijen zullen deze garanties moeten handhaven om zinvol toegepast te kunnen worden. Dit stelt eisen aan de inrichting en eventueel intern transport. Dit geldt met name voor de betrouwbare profielen. |
| Traceerbaarheid | De berichtenstroom is traceerbaar via elke schakel in de logistieke keten.  | Elke schakel in de Digikoppeling-keten moet inkomende en uitgaande berichten monitoren, loggen en moet voorzien in een audittrail.  Dit geldt met name voor de betrouwbare profielen.  |
| Foutafhandeling  | Fouten worden correct en tijdig afgehandeld. Uitval van meldingen wordt zoveel mogelijk voorkomen, mede door het gebruik van een betrouwbaar profiel.  | Elke schakel in de Digikoppeling-keten moet foutafhandeling inrichten.  Dit geldt met name voor de betrouwbare profielen. |


Tabel 6.2: Digikoppeling-voorschriften

## REST API's

Het [[[DK-RESTAPI]]] is gebaseerd op de Federated Services Connectivity (FSC) standaard en de REST API Design Rules die in 2020 door het Kennisplatform API's zijn ontwikkeld.

Een application programming interface (API) is een gestructureerd en gedocumenteerd koppelvlak voor communicatie tussen applicaties. In de laatste 10 jaar heeft *REpresentational State Transfer* (REST) zich ontwikkeld tot een bepalend principe voor het realiseren van API's.

De standaard FSC schrijft voor hoe gekoppeld kan worden met een API's, hoe API's in een netwerk gevonden kunnen worden en wanneer en hoe log regels moeten worden weggeschreven. 

De standaard REST API Design Rules geeft een verzameling basisregels voor structuur en naamgeving waarmee de overheid op een uniforme en eenduidige manier REST API's aanbiedt. Dit maakt het voor ontwikkelaars gemakkelijker om betrouwbare applicaties te ontwikkelen met API's van de overheid. REST API's kunnen worden gebruikt voor het laagdrempelig bevragen van resources maar ook voor het creëren en muteren van resources.

### Digikoppeling REST API voor synchrone requests

[[[DK-RESTAPI]]] biedt de volgende functionaliteiten:

- Vertrouwelijkheid
- Identificatie en authenticatie van partijen
- Versleuteling op basis van mTLS conform de Digikoppeling Beveiligings voorschriften
- Mechanisme voor het autoriseren van koppelingen met een API
- Mechanisme voor het ontdekken van API's op een netwerk
- Delegatie: een API aanbieden of consumeren namens een andere organisatie
- Logging van verzoeken naar API's
- (Status)Responsecodes en Foutmeldingen

### OAS: OpenAPI Specification

Een [[[openapi]]] beschrijft de eigenschappen van de data die een API als input accepteert en als output teruggeeft. OAS specificeert alleen welke attributen de API verwerkt en hun datatypen, niet welke implementatie er achter de API schuilgaat.

Voor het beschrijven van DK-Rest API's is het gebruik van OAS verplicht. Op [[Pas-toe-of-leg-uit]] staat beschreven welke versie toegepast moet worden.

## WUS

### WUS familie van standaarden

Digikoppeling maakt gebruik van een familie van standaarden die we binnen Digikoppeling de naam “WUS” geven. Deze familie van standaarden is gebaseerd op webservice standaarden uit de profielen van de OASIS “Web Services – Basic Reliable and Secure Profiles” Technical Committee (WS-BRSP)<sup>[27](#f27)</sup>. De naam WUS staat voor WSDL, UDDI en SOAP, drie belangrijke deelstandaarden. Hoewel Digikoppeling geen gebruik van UDDI maakt is deze term inmiddels gebruikelijk.

Kenmerkend voor de WUS-standaarden die voortkomen uit de Internet-wereld is de 1-op-n relatie tussen service aanbieder en meerdere service afnemers. Dit betekent b.v. dat een WUS service één WSDL heeft die door alle afnemers kan worden gebruikt.

<sup><a name="f27"><dfn>27</dfn></a>: Voorheen Web Services Interoperability (WS-I) organization</sup>

### Digikoppeling WUS voor synchrone bevragingen

De [[[DK-WUS]]] ondersteunt het uitvoeren van synchrone requests tussen geautomatiseerde informatiesystemen.

[[[DK-WUS]]] biedt de volgende functionaliteiten: 

- Identificatie en authenticatie van partijen
- Versleutelen van transport
- Adresseringsinformatie voor routering ‘achter de voordeur’
- Routeren via message-handlers
- berichtuitwisseling vast leggen in standaard technisch contract formaat
- Beveiligen van berichten d.m.v. technische handtekening
- Beveiligen van berichten door de content te versleutelen
- Foutmeldingen

### WSDL: Web Services Description Language

Een WSDL is een formeel xml-document om de gebruikte functionele en technische eigenschappen van de (XML-)berichtuitwisseling via WUS vast te leggen. Elke service heeft één WSDL, die door de serviceaanbieder wordt opgesteld. Deze is door alle afnemers te gebruiken. Door importeren van de WSDL in de Digikoppeling-adapter van een afnemer wordt de berichtuitwisseling geconfigureerd.

De wijze waarop een WSDL wordt toegepast staat beschreven in Digikoppeling Best Practices WUS.

## ebMS

### ebMS2 familie van standaarden

Digikoppeling maakt  gebruik van een familie van standaarden die we “ebMS2” noemen. Deze familie van standaarden is gebaseerd op web-service standaarden uit de profielen van de OASIS “ebXML Messaging Services“ Technical Committee (ebMS2).

Kenmerkend voor de ebMS2-standaarden die voortkomen uit de EDIFACT-wereld is de 1-op-1 relatie tussen een beperkt aantal (vaak twee) partijen. Dit betekent dat twee partijen samen een CPA moeten afspreken, creëren en implementeren; de CPA is dus van zowel de serviceaanbieder als de serviceafnemer.

### Digikoppeling ebMS2 voor betrouwbare, asynchone uitwisseling  

De [[[DK-ebMS]]] ondersteunt het uitvoeren van asynchrone berichten tussen geautomatiseerde informatiesystemen.

Het protocol regelt de betrouwbare ontvangst van een bericht en eventueel de onweerlegbaarheid (non-repudiation) in de vorm van een ondertekende ontvangstbevestiging. Hoewel Digikoppeling-meldingen (op de logistieke laag) asynchroon zijn kan de business-laag wel synchroon werken als de verzender wacht op een retourmelding.`

De Koppelvlakstandaard ebMS2 regelt de volgende functionaliteiten: :

- Identificatie en authenticatie van partijen
- Versleutelen van transport
- Adresseringsinformatie voor routering ‘achter de voordeur’
- Routeren via message-handlers
- Asynchroon berichten correleren d.m.v. message ID
- Meerdere berichten logisch samenvoegen
- Berichten voorzien van een beveiligde datum en tijdstempel (time-stamping)
- Berichtuitwisseling vast leggen in standaard technisch contract formaat (servicecontract)
- Beveiligen van berichten d.m.v. technische handtekening
- Beveiligen van berichten door de content te versleutelen
- Onweerlegbaarheid op protocolniveau (non-repudiation)
- Betrouwbaar asynchroon berichten versturen met ontvangstbevestigingen
- Ondersteuning voor foutafhandeling op asynchrone berichten
- Volgorde van berichten zo mogelijk handhaven
- Hertransmissies op protocolniveau totdat ontvangst is bevestigd

### CPA

Een CPA is een formeel xml-document om de gebruikte functionele en technische eigenschappen van de ebMS2 protocol-karakteristieken vast te leggen. Het is dus een formele beschrijving voor het vastleggen van de gegevensuitwisseling. Een CPA moet worden gecreëerd als twee partijen afspreken om van elkaars ebMS2 services gebruik te maken. Beide partijen moeten de CPA importeren in hun Digikoppeling-adapter om deze te configureren voor de berichtuitwisseling.

De wijze waarop een CPA wordt toegepast staat beschreven in Digikoppeling Best Practices ebMS2. Het CPA Register ondersteunt partijen in het creëren van een CPA.

## Grote berichten

### Werking grote berichten

De situatie kan zich voordoen dat een Digikoppelingbericht een grootte krijgt die niet meer efficiënt door de Digikoppelingadapters en -services verwerkt kan worden. Ook kan er behoefte zijn aan het buiten de normale procesgang ('out-of-band') sturen van aanvullende informatie naar systemen. In die gevallen zal dit “grote bericht” op een andere wijze verstuurd moeten worden: middels de Digikoppeling koppelvlakstandaard Grote Berichten.

De volgende standaard aanpak wordt hierbij gehanteerd:

- Met WUS, ebMS2 of eventueel REST wordt referentie (link) verstuurd;

- de referentie verwijst naar de locatie van het grote bestand. Het hangt af van het  gebruikte Digikoppeling Grote berichten profiel of de ontvanger het bestand moet downloaden of dat de zender het grote bestand inmiddels als naar de ontvanger heeft geupload.

Het grote bericht zelf zal vaak volledig in het grote bestand zijn opgenomen; het WUS, ebMS2 of REST-bericht bevat dan alleen metadata (waaronder de link naar het bestand). Maar het kan ook gebeuren dat een klein deel van het oorspronkelijk grote bericht al in het WUS-bericht is opgenomen en de rest (bijvoorbeeld bijlagen bij het bericht) in een of meerdere bestanden is opgenomen.

Het principe dat Digikoppeling grote berichten toepast is het ‘claim-check’ principe. Dit betekent dat het bericht zelf (WUS/ebMS2/REST) alleen een referentie (claim-check) naar het grote bestand bevat. Deze referentie wordt vervolgens gebruikt om het bestand zelf op te halen.

Een belangrijk voordeel hiervan is dat het grootste deel (het grote bestand zelf) de berichtenuitwisseling niet verstoort doordat het niet door de message-handler afgehandeld hoeft te worden (en deze bijvoorbeeld vertraagt). Maar ook is een voordeel dat de afhandeling van het grote deel op een ander moment in de tijd kan plaatsvinden en daardoor de procesgang van achterliggende informatiesystemen niet verstoord.

De standaard doet geen uitspraak over gegevensstromen waarin kleine en grote berichten voorkomen. Bij implementatie van dergelijke gegevensstromen zal een organisatie moeten afwegen of kleine berichten anders of gelijk aan de ‘echte’ grote berichten verwerkt worden. In z’n algemeenheid zal een uniforme afhandeling eenduidiger en vooral ook eenvoudiger zijn; slechts in bijzondere gevallen zal dit niet volstaan.

### Standaarden voor grote berichten

De [[[DK-GB]]] maakt gebruik van WUS, ebMS2 of REST
 voor het verzenden van metadata. Voor ophalen van het grote bestand maakt de standaard gebruik van HTTPS-downloads. Daardoor zijn reliability en security gelijkwaardig aan de andere koppelvlakstandaarden. Ook is het gebruik van transparante intermediairs mogelijk.

[[[DK-GB]]] regelt de volgende functionaliteiten, in aanvulling op WUS of ebMS2

- Identificatie en authenticatie van partijen (OIN)

- Versleutelen van transport

- Routeren via (http) proxies

- Bestand correleren aan bericht

- Ondersteuning voor foutafhandeling

- Na onderbreking hervatten waar de overdracht is afgebroken (‘resume’)

- Optioneel beperkte tijdsperiode om bestand beschikbaar te stellen.
