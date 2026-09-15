# Danske personabeskrivelser

Du skriver syv selvstændige beskrivelser af den samme opdigtede danske person. Hver
beskrivelse viser, hvordan profilen og de godkendte attributter tilsammen skaber ét
bestemt menneske med sit eget perspektiv.

Returnér kun data, som passer til det krævede JSON-skema. Hver tekst skal være på 2-4
sætninger, konkret uden at være identificerende og indbyrdes forskellig.

Regler:

- Skriv i nutid og konstaterende. Brug ikke forbehold som "muligvis", "måske",
  "kan tænkes", "har lettere ved", "formentlig", "sandsynligvis" eller "en tendens til".
- Personen er opdigtet, så beskriv hvad hun eller han gør, ikke hvad der kunne passe.
- Bevar demografi, personlighed og de allerede godkendte attributter.
- Brug altid alderen aktivt i beskrivelsen.
- Oversæt personlighedsprofilen til konkret adfærd. Gengiv aldrig scorer, etiketter
  eller psykologiske vurderinger, og beskriv ikke et træk som en mangel eller en
  begrænsning.
- Væv den kulturelle kontekst ind implicit i stedet for at nævne den direkte.
- Alle syv domæner skal beskrives konkret. Fylder et domæne lidt i personens liv, så
  skriv hvordan det konkret indgår alligevel, frem for at skrive at det ikke passer.
- Hold hver specialiseret tekst til sit eget domæne; sportsafsnittet må eksempelvis ikke
  fyldes med kunstinteresser.
- Konkrete, individuelle detaljer er ikke stereotyper. Undgå træk, der blot følger
  kategorisk af køn, alder, region, uddannelse eller arbejdsstatus.
- Herkomst er et fast input fra Danmarks Statistik med fem kategorier: dansk
  oprindelse, indvandrer eller efterkommer fra henholdsvis vestlige og ikke-vestlige
  lande. Det er en administrativ kategori, ikke etnicitet, religion eller nationalitet.
- Lad herkomsten forme personen implicit gennem hverdagen, fx sprog i hjemmet,
  familie i udlandet eller madtraditioner. Nævn aldrig kategorien direkte.
- Opfind aldrig et konkret oprindelsesland, en landegruppe eller et sprog ud over det,
  inputtet angiver. "Vestlige" og "ikke-vestlige" dækker mange lande, og et gæt er
  udokumenteret.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke religion, etnicitet, helbred, seksualitet, politisk overbevisning eller
  kriminalhistorik.
- Gengiv højst den angivne region; opfind ikke bystørrelse, lokale tilbud, rejsevaner
  eller andre stedsegenskaber.
- Brug naturlige danske ord frem for unødige engelske sammensætninger.
- `persona` skal samle helheden uden at gentage de fem specialiserede tekster ordret.

## `visual_persona`

Dette felt bruges som grundlag for et portrætbillede, så det følger andre regler end de
øvrige seks. Skriv det som en neutral billedbeskrivelse i tredje person, ikke som en
fortælling.

- Begynd med køn og alder, og skriv dem eksplicit: "Kvinde, 20 år."
- Skriv derefter oprindelsen ud fra feltet `origin_region`. Er den `danmark`, skal
  sætningen udelades helt. Ellers bruges præcis denne formulering, hvor `<sted>` kommer
  fra tabellen nedenfor:
  - indvandrer: "Hun er indvandret til Danmark fra <sted>." eller hankønsformen.
  - efterkommer: "Hun er født i Danmark af forældre fra <sted>." eller hankønsformen.

| `origin_region` | `<sted>` |
| --- | --- |
| `vesteuropa_og_eu` | et andet vesteuropæisk land |
| `europa_uden_for_eu` | et land i Europa uden for EU |
| `mellemosten_og_nordafrika` | Mellemøsten eller Nordafrika |
| `asien` | et land i Asien |
| `afrika_syd_for_sahara` | Afrika syd for Sahara |
| `latinamerika_og_caribien` | Latinamerika eller Caribien |
| `nordamerika_og_oceanien` | Nordamerika eller Oceanien |
| `ovrige` | et andet land |

- Gæt aldrig på et konkret land, en hudfarve, en etnicitet eller en religion. Gruppen
  ovenfor er det eneste, der må siges om oprindelse, og den dækker mange lande.
- Beskriv derefter kun det, et portræt viser: hår, briller, skæg, tøj øverst på kroppen
  og et roligt, neutralt ansigtsudtryk. Hold det til to eller tre konkrete detaljer.
- Disse detaljer er opdigtede ligesom interesser og færdigheder. De må passe til alder,
  arbejdsmarkedsstatus og personlighed, men de er ikke udledt af oprindelsen.
- Skriv ikke om kroppen ud over hovedet og skuldre, og beskriv ikke skønhed,
  tiltrækningskraft, vægt eller helbred.
- Nævn ikke kamera, lys, baggrund eller billedformat. Det tilføjer billedleddet selv.
