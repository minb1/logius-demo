
# Beveiliging, autorisatie en protocollen

## Authenticatie en autorisatie

Binnen de Digimelding-keten vindt op meerdere plaatsen authenticatie en
autorisatie plaats.

In deze sectie wordt uitgelegd welke verschillende middelen er zijn en
in welke stap ze gebruikt worden. Binnen Digimelding vinden
authenticatie en autorisatie plaats op transportniveau, applicatieniveau
en soms op persoonsniveau. Transport en logistiek gaan via de
Digikoppeling-standaarden.

Op applicatieniveau gaat het over de terugmeldapplicatie van de afnemer,
Digimelding Webservice en TMVs van basisregistraties.

Persoonsniveau komt voor bij Digimelding Portaal en mogelijkerwijs bij
nog te ontwikkelen andere oplossingen die gebruik maken van eHerkenning
of SSOn Rijk. Daarnaast kent een afnemende organisatie ook nog interne
authenticatie en autorisatie voor het verkrijgen van toegang door de
terugmelder tot de terugmeldapplicatie. Dit is buiten scope van deze
specificatie.

![Authenticatie en autorisatie bij Digimelding](images/image2.png "Authenticatie en autorisatie bij Digimelding")

| **Autorisatie** |                |                |                |
|     :---        |    :---        |    :---        |    :---        |
| **Terugmelder** | **Terugmeld Applicatie**   | **Digikoppeling Adapter** | **TMV Basisregistratie** |
| E-herkenning in DMKS bericht<br>OIN<br>Vestigingsnr<br>Persoonsaanduiding | OIN in DMKS bericht | OIN in Digikoppeling Headers WS-Adressing | Transport via Digikoppeling |
| SSOn Rijk<br>OIN<br>Vestigingsnr<br>Persoonsaanduiding | (PKIO certificaat met OIN voor signing) |  PKIO certificaat met OIN voor TLS/signing | Applicatie: via OIN in DMKS bericht of via E-Herkenning : OIN, vestigingsnummer en persoonsaanduiding in DMKS bericht of via SSOn Rijk attributen |
| Vrije keuze | OIN in Digikoppeling Header WS-security | | |

Figuur 2. Authenticatie en autorisatie bij Digimelding

Indien in de keten de terugmelder met eHerkenning geauthenticeerd wordt,
willen basisregistraties dit altijd gebruiken als autorisatiemiddel en
dienen de eHerkenningsgegevens ( 'OIN'<sup>1</sup>, 'vestigingsnummer' en
'pseudoID' van de natuurlijk persoon(ontvangen via een SAML-token) in
het DMKS-bericht doorgegeven te worden. In deze gevallen is de
autorisatie voor de TMV-applicatie geregeld middels eHerkenning op
persoonsniveau.

<p class="note">
<sup>1</sup> De eHerkenningmiddelenleverancier stuurt een SAML token terug met daarin
o.a. het 'OIN' zoals in de eHerkenningskoppelvlakstandaard is
gespecificeerd. Dit 'OIN' met een prefix 00000003 is niet gelijk aan
het door Logius uitgegeven OIN zoals in het OIN-register is
opgenomen.</p>

Indien in de keten de terugmelder met SSOn-Rijk geauthenticeerd wordt,
willen basisregistraties dit altijd gebruiken als autorisatiemiddel en
dienen de gegevens ('OIN', en 'pseudoID' ) van de natuurlijk persoon
(ontvangen via een SAML-token) in het DMKS-bericht doorgegeven te
worden. In deze gevallen is de autorisatie voor de TMV-applicatie
geregeld middels SSOn-Rijk op persoonsniveau.

In het geval dat er geen authenticatie op Persoonssniveau plaatsvindt,
worden authenticatie en autorisatie op applicatieniveau gedaan op basis
van het OIN van de afnemende organisatie. Dit is het OIN dat een
organisatie krijgt wanneer deze zich aanmeldt voor Digikoppeling en is
terug te vinden in het OIN-register van Logius. Dit OIN wordt opgenomen
in het DMKS-bericht.

Indien de basisregistratie er om vraagt kan als extra
authenticatiemiddel signing van het DMKS-bericht met een
PKIoverheidcertificaat worden gevraagd. Dit certificaat dient van de
afnemende organisatie te zijn en het OIN te bevatten zoals uitgegeven
toen de organisatie zich aanmeldde voor Digikoppeling.

Op transportniveau wordt geautoriseerd op het OIN dat door de
Digikoppeling-adapter van de afnemende organisatie wordt gebruikt. Dit
wordt gebruikt in de Digikoppeling headers WS-Addressing (WUS) en
PartyID (ebMS) en in het PKIoverheidscertificaat dat voor transport
beveiliging (TLS) wordt gebruikt. Dit is niet altijd het OIN van de
organisatie zelf! Bij sectorale knooppunten, SaaS-oplossingen en
samenwerkingsverbanden kan dit mogelijk een ander OIN zijn. De interne
authenticatie en autorisatie (indien anders dan eHerkenning/SSOn-Rijk)
is volledig vrij door de afnemende organisatie in te vullen. Hiervan
hoeft niets meegegeven te worden in DMKS-berichten.

### Protocollen

Bij synchrone communicatie wordt gebruik gemaakt van Digikoppeling WUS
volgens het '2W-be'-profiel<sup>2</sup> met ondersteuning voor MTOM voor
efficiënt transport van attachments.

Indien een basisregistratie verzoekt om een authenticatie van de
terugmeldende organisatie wordt bij synchrone communicatie gebruikt
gemaakt van WUS Profiel Digikoppeling 2W-be-S<sup>3</sup>.

<p class="note">
<sup>2</sup> Best Effort, beveiligd met tweezijdige TLS
</p>
<p class="note">
<sup>3</sup> Best Effort, beveiligd met tweezijdige TLS en gesigneerde berichten
</p>
