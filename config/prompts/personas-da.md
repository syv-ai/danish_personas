# Danske personabeskrivelser

Du skriver syv korte, syntetiske personabeskrivelser på naturligt dansk ud fra faste
strukturerede input.

Returnér kun data, som passer til det krævede JSON-skema. Hver tekst skal være på
2-4 sætninger, konkret uden at være identificerende og indbyrdes forskellig.

Regler:

- Bevar demografi, personlighed og de allerede godkendte attributter i de
  relevante beskrivelser, men gengiv dem ikke i `visual_persona`.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke hudfarve, etnicitet, herkomst, nationalitet, kultur, sprog, religion,
  helbred, handicap, seksualitet, politisk overbevisning, kropsmål, udseende,
  tiltrækningskraft eller kriminalhistorik.
- `visual_persona` skal være 2-4 sætninger og må kun beskrive valgte, foranderlige
  stilvalg: tøj, tilbehør og farver samt et generisk, ikke-identificerende
  portrætmiljø som en neutral baggrund eller et atelier. Det er ikke
  billedgenerering.
- `visual_persona` må hverken gentage eller udlede input om alder, køn, sex,
  region, uddannelsesniveau, civilstand eller arbejdsmarkedsstatus. Den må heller
  ikke gentage eller udlede OCEAN-scorer, genererede attributter, oprindelse,
  herkomst, nationalitet eller sprog.
- `visual_persona` må ikke beskrive fysiske, uforanderlige eller følsomme træk,
  herunder hud, hår, øjne, krop, højde, vægt, helbred, handicap, seksualitet,
  religion, politik, etnicitet, tiltrækningskraft eller identitet. Vælg ikke
  detaljer eller miljø ud fra nogen af disse oplysninger.
- Undgå stereotyper og deterministiske forbindelser mellem demografi og personlighed.
- Behandl OCEAN-scorer som svage tilbøjeligheder, ikke som fakta om evner, problemer
  eller begrænsninger. Brug afbalancerede formuleringer som "kan foretrække" og
  "har muligvis lettere ved", og beskriv aldrig en lav eller høj score som en mangel.
- Undlad at gengive scoreetiketter eller psykologiske vurderinger i teksten.
- Hvis et domæne ikke passer naturligt, skal teksten sige det neutralt frem for at
  opfinde ekspertise.
- Hold hver specialiseret tekst til sit eget domæne; sportsafsnittet må eksempelvis
  ikke fyldes med kunstinteresser.
- Omtal syntetiske færdigheder som interesser eller mulige styrker, ikke som
  dokumenteret erfaring eller konkrete arbejdsopgaver.
- Gengiv højst den angivne region i de relevante ikke-visuelle beskrivelser; opfind
  ikke bystørrelse, lokale tilbud, rejsevaner eller andre stedsegenskaber. Nævn ingen
  region eller placering i `visual_persona`.
- Brug naturlige danske ord frem for unødige engelske sammensætninger.
- `persona` skal samle helheden uden at gentage de fem specialiserede tekster ordret.
