# CPA Gebruik en Kenmerken

## Inleiding

Digikoppeling bevat de ebMS2 koppelvlakstandaard voor de overdracht van gegevens. Dit document gaat in op het gebruik van een Collaboration Protocol Agreement (CPA) in het geval dat ebMS2 gebruikt wordt voor de gegevensoverdracht.

## Wat is een CPA?

Een CPA is een formeel xml document om de gebruikte functionele en technische eigenschappen van de ebMS2 protocol-karakteristieken vast te leggen. Het is dus een formele beschrijving voor het vastleggen van de gegevensuitwisseling.

De CPA is gestandaardiseerd in [ISO 15000-1: ebXML Collaborative Partner Profile Agreement (afgekort tot ebCPP<sup>3</sup>). Het ebMS2 protocol is gestandaardiseerd in ebXML Messaging Service Specification (afgekort tot ebMS2<sup>4</sup>).

> <sup>3</sup>. [[ebCPP]] Collaboration-Protocol Profile and Agreement Specification Version 2.0, September 23, 2002. Url: http://www.oasis-open.org/committees/ebxml-cppa/documents/ebcpp-2.0c.pdf
> <sup>4</sup>. Message Service Specification, Version 2.0, 1 April 2002. Url: http://www.oasis-open.org/committees/ebxml-msg/documents/ebMS_v2_0.pdf [[EBXML-MSG]]

De eigenschappen van de gegevensoverdracht geven onder andere aan:

- tussen welke partijen er informatie uitgewisseld wordt;

- welke services en actions ('functies') er zijn waar de berichtuitwisseling op wordt gebaseerd;

- hoe certificaten gebruikt worden voor bijvoorbeeld transportbeveiliging, payload encryptie en/of ondertekening van berichten;

- wat de overdrachts karakteristieken zijn, zoals de intervallen voor hertransmissie als betrouwbare overdracht gewenst is;

- hoe om te gaan met acknowledements;

- wat de eigenschappen zijn van de transportkanalen;

## Waarom wordt er een CPA gebruikt?

Redenen voor het toepassen van een CPA:

- het is een formeel contract tussen twee partijen, die op basis van ebMS2 gegevens willen uitwisselen;

- het automatiseert de e-configuratie van de ebMS-adapter (het inlezen van de CPA volstaat);

- het biedt de zekerheid dat beide partijen dezelfde instellingen gebruiken;

Daarom is het hebben van een CPA het uitgangspunt voor de specificatie en configuratie van de gegevensuitwisseling tussen twee partijen op Digikoppeling.

## Wat zijn de uitgangspunten voor de CPA?

De kenmerken van het ebMS2 verkeer op Digikoppeling zijn beschreven in het document:

' Digikoppeling Koppelvlakstandaard ebMS2'. Dit document is op de website van Digikoppeling te vinden [[?Digikoppeling Logius website]].

De kenmerken zijn vertaald naar relevante onderdelen van een CPA. Deze CPA onderdelen worden hieronder beschreven in termen zoals benoemd in [[ebCPP]].

### Service

Een Service is een unieke aanduiding voor een set van samenhangende operaties.

De service moet uniek zijn. Advies voor de naamgeving is als volgt:

`[organisatie]:[service]:[versie]`

De service heeft als type “urn:osb:services” (zonder quotes).

Een service wordt als volgt in het ebMS2 contract opgenomen (met een voorbeeld service “OSB:Service:1:0”):  

```XML
<tns:Service tns:type="urn:osb:services">OSB:Service:1:0</tns:Service>
```

### CPAId

Een CPAId is een unieke aanduiding voor een overeenkomst voor een bepaalde service. De CPAId moet uniek zijn voor die bepaalde service. Denk hierbij aan het feit dat er één service wordt gespecificeerd door een service provider en dat meerdere service requesters een samenwerking aangaan. Elke samenwerking tussen twee partijen moet een ander CPAId krijgen. Een CPAId moet uniek zijn. Advies voor de naamgeving is als volgt:

`[ServiceID]_[PartyId-A]_[PartyId-B]_[versie]`

NB. Gebruik geen punten of andere vreemde tekens, want sommige ebMS-adapters zouden de CPAId wel eens als filenaam kunnen gebruiken...

Hierbij zijn:

  - `[ServiceID]` de unieke Service ID

  - `[PartyId-A]` de PartyId van partij A

  - `[PartyId-B]` de PartyId van partij B

  - `[versie]` een UUID (of een versie nummer mits de uitgever van de CPA kan garanderen dat de CPAId uniek blijft)

### Start- en einddatum

Elke CPA heeft een start en einddatum. Dit moet afgestemd worden tussen de twee partijen die een samenwerking aangaan. Merk op dat er een afhankelijkheid is met de geldigheidsperiode van de gebruikte client- en servercertificaten.

### Default Message Channel

Een CPA bevat twee PartyInfo elementen: voor elke deelnemer in de samenwerking één. Elk PartyInfo element kent precies één 'default channel' dat gebruikt wordt voor de verzending van onderliggende protocol berichten (zoals de acknowledgments). In de CPA wordt deze 'default channel' aangegeven met het defaultMshChannelId attribuut. De eigenschappen van dit channel worden bepaald op basis van het Digikoppeling ebMS2 profiel met de hoogste beveiliging. Als een CPA verschillende Actions bevat waarvoor de acknowledgements verschillende profiel eigenschappen hebben, zullen de Actions verdeeld moeten worden over meerdere CPA's: in elke CPA komen die Actions die dezelfde profiel eigenschappen hebben. Als er gebruik gemaakt wordt van de CPA Creatievoorziening zullen er verschillende Digikoppeling-ebMS2 Servicespecificaties gemaakt moeten worden.

<aside class="example">
Er zijn twee Actions:  
Action1 : profiel osb-rm-e  
Action2 : profiel osb-be  
De default channel zal de eigenschappen overnemen van het profiel osb-rm-e. Als dit NIET wenselijk is, zullen de twee actions in twee verschillende CPA's geplaatst moeten worden.
</aside>

### PartyName

De naam van de partij zoals die opgegeven moet worden in de CPA. Dit zal voor elke organisatie anders zijn, maar blijft wel hetzelfde voor alle CPA’s die gemaakt zullen worden.

### PartyId

Het Organisatie Identificatie nummer (OIN) van de organisatie. De PartyId is de (logische) aanduiding waarmee de organisatie geïdentificeerd wordt. Als de organisatie nog geen OIN heeft moet een nummer aangevraagd worden bij Logius. De COR (De Centrale OIN Raadpleegvoorziening) wordt gebruikt om informatie vast te leggen over de organisatie en het OIN. Het nummer zal ook gebruikt worden in het cliënt certificaat voor het subject.Serialnumber veld. Zie ook EB014 in hoofdstuk 2. Organisaties die op basis van standaarden met andere overheden communiceren wordt sterk aangeraden om een OIN aan te vragen.

### PartyId Type

Deze heeft de waarde `urn:osb:oin` voor PartyId's met een OIN.(Dit is ook de default waarde voor de CPA's zoals die door het CPA register wordt gehanteerd.)

De PartyId type wordt als volgt opgenomen in het ebMS2 contract (met een voorbeeld van de PartyId waarde `0123456789`):  

```XML
<tns:PartyId tns:type="urn:osb:oin">123456789</tns:PartyId>
```

Het is toegestaan om een andere PartyId type te hanteren als de organisatie reeds andersoortige (geen OIN's) PartyId’s heeft voor de organisatie identificatie. Het moge duidelijk zijn dat het in overleg met de samenwerkende organisaties vastgesteld moet worden. Zie ook EB014 in hoofdstuk 2.

### BusinessCharacteristics

Deze heeft de volgende verplichte waarde, waarbij alleen de timeToPerform een andere waarde kan krijgen (afhankelijk van de timing karakteristieken van de RequestingBusinessActivity en de RespondingBusinessActivity):

```XML
<tns:BusinessTransactionCharacteristics
    tns:isAuthenticated="transient"
    tns:isAuthorizationRequired="true"
    tns:isConfidential="transient"
    tns:isIntelligibleCheckRequired="false"
    tns:isNonRepudiationReceiptRequired="false"
    tns:isNonRepudiationRequired="false"
    tns:isTamperProof="transient"
    tns:timeToPerform="P2D"/>
```

### TransportProtocol over HTTP met TLS met server certificaat
Deze hebben de verplichte waardes:

```XML
<tns:TransportProtocol tns:version="1.1">HTTP</tns:TransportProtocol>
```

en voor bijvoorbeeld versie 1 van TLS

```XML
<tns:TransportSecurityProtocol tns:version="1.0">TLS</tns:TransportSecurityProtocol>
```

<aside class="note">
voor de actuele versies van het te gebruiken protocol bij de uitwisseling zie [[Digikoppeling Beveiligingsdocument]]
</aside>

### Client Authentication over HTTP met client certificaat. 

Dit is verplicht. In het client certificaat staat in het subject.Serialnumber de PartyId van de ‘client’ organisatie.

### Endpoint

Deze heeft de waarde van de URL (FQDN, met pad namen) van de ebMS-adapter waarmee over Digikoppeling gegevens uitgewisseld worden. De FQDN van de URL moet overeenkomen met de FQDN die in het server certificaat vermeld staat.

### MessageOrder

De MessageOrder geeft aan of er wel of geen gebruik gemaakt wordt van ordening van berichten. De default waarde voor MessageOrder is “NotGuaranteed” en wordt als volgt opgenomen in het ebMS2 contract:

```XML
    <tns:MessageOrderSemantics>NotGuaranteed</tns:MessageOrderSemantics>
```

Indien er wel gebruik gemaakt wordt van MessageOrder is de waarde:

```XML
    <tns:MessageOrderSemantics>Guaranteed</tns:MessageOrderSemantics>
```

<aside class="note"> MessageOrder wordt niet door alle ebMS-adapters implementaties ondersteund. Als het wel het geval is zal de interoperabiliteit goed getest moeten worden. Zie hoofdstuk “Het gebruik van bericht volgordelijkheid”.
</aside>

### ReliableMessaging

Deze heeft default een retryCount van 8 en een retryInterval van 3 uur, zonder MessageOrder:

```XML
<tns:MessagingCharacteristics tns:syncReplyMode="none"
tns:ackRequested="always" tns:actor="urn:oasis:names:tc:ebxml-msg:actor:toPartyMSH"
tns:ackSignatureRequested="never"    tns:duplicateElimination="always"/>
```

De waardes kunnen per CPA bepaald worden, en liggen dus niet bij voorbaat vast.

In het geval dat MessageOrder wel gebruikt wordt, komt in de CPA:

<tns:MessageOrderSemantics>Guaranteed</tns:MessageOrderSemantics>

Conform de ebMS2 specificatie zal de applicatie dezelfde ConversationId moeten gebruiken voor de opeenvolgende berichten<sup>5</sup>.

> <sup>5</sup>. [[EBXML-MSG]] H9.1.1 “The REQUIRED SequenceNumber element indicates the sequence a Receiving MSH MUST process messages. The SequenceNumber **is unique within** the ConversationId and MSH.”

### PersistDuration

Deze heeft default de waarde van 1 dag, maar zal anders zijn als er andere waardes voor ReliableMessaging gebruikt worden:

```XML
<tns:PersistDuration>P1D</tns:PersistDuration>
```

### MessagingCharacteristic

Deze heeft de waarde:

```XML
<tns:MessagingCharacteristics

tns:syncReplyMode="none"

tns:ackRequested="always"

tns:actor="urn:oasis:names:tc:ebxml-msg:actor:toPartyMSH"

tns:ackSignatureRequested="never"

tns:duplicateElimination="always"/>
```

Geen gebruik van Signing of (payload) Encryption. (Alleen op HTTP nivo wordt informatie beveiligd)

### Certificaten

Er moet gebruik gemaakt worden van certificaten die voldoen aan de eisen van de PKI Overheid.

## Hoe wordt een CPA gemaakt?

Op basis van de hierboven genoemde CPA onderdelen kan alleen een 'CPA template' gemaakt worden. Wat ontbreekt zijn de specifieke zaken rondom:

- De services en functies (Actions in ebMS2 terminologie) die aangesproken kunnen worden, inclusief procesnaam waar ze deel van uitmaken.

- De technische gegevens van een ebMS-gateway van een organisatie, zoals de te hanteren transport URL, de publieke sleutels van de client en server certificaten ende PartyId van de organisatie

In het document 'Digikoppeling CPA Creatiehandleiding' is te lezen met welke gegevens een CPA gemaakt wordt, in combinatie met Digikoppeling CPA Creatievoorziening.

| Meer informatie     | Zie document in de aansluitkit               | Doelgroep |
|-------------------------|--------------------------------------------------|---------------|
| Handleiding CPA register | [https://www.logius.nl/diensten/digikoppeling/aanvragen/voorbereiden](https://www.logius.nl/diensten/digikoppeling/aanvragen/voorbereiden) | [A&D] [OT&B]  |

