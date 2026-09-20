# Danske personabeskrivelser

Du skriver seks korte, syntetiske tekster på naturligt dansk ud fra faste input:
`professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`,
`culinary_persona` og `persona`. Returnér kun JSON efter skemaet. Hver tekst er
sammenhængende prosa på 2-4 sætninger. `persona` er højst 600 tegn og er beregnet til
senere billedprompting; den må ikke være en visuel beskrivelse.

`persona` skal indeholde hvert naturligt sætningsled i `required_persona_facts` ordret
og som en del af løbende prosa. Det gælder alle fem sætningsled: pronomen og alder,
bopælskommune, oprindelse, uddannelse og nuværende beskæftigelse. Du må kun ændre det
første bogstav til stort, når et sætningsled står først i en sætning. Oversæt, bøj,
uddyb eller pynt aldrig på sætningsleddene, og skriv dem ikke som en liste.

Brug altid det leverede pronomen. Statistisk køn må kun fremgå af det krævede
sætningsled med `han` eller `hun`. Brug aldrig `mand` eller `kvinde` som selvstændige
ord, heller ikke i bøjet form. Skriv altid `kommer fra X`; brug aldrig tekniske
betegnelser som
`oprindelsesland` eller `oprindelsesetiket`. Brug den leverede formulering `arbejder
som X` for en ansat og `er X` for en anden aktuel status. Uddannelsesformuleringen er
et midlertidigt, kildeunderbygget, sammenlagt niveau, indtil en særskilt DST-kilde kan
give flere detaljer. Udled eller opfind aldrig en bestemt uddannelse eller en
uddannelsesinstitution. Brug ingen ord fra datamodellen eller andre tekniske
forklaringer i teksten.

Returnér altid alle skemafelter; ingen felter må udelades. Indlejr nøjagtigt 2-3
forskellige interesser ordret fra `generated_attributes` i løbende prosa. Bevar små
bogstaver; kun en interesse først i en sætning må begynde med stort bogstav. Medtag
nøjagtigt 1-2 forskellige OCEAN-tendenser som komplette, bogstavelige fraser fra den
leverede liste `allowed_personality_tendencies`. Kopiér hver hel frase ordret;
sammensæt aldrig en term med en separat forsigtighedsmarkør, og brug ikke andre
personlighedstermer. Rå OCEAN-scorer og etiketter hører ikke til dette trin: fortolk
dem ikke, og opfind ikke andre personlighedstermer end de leverede fraser. Skriv
aldrig punktlister, nummerering, parenteser, klammer eller semikolonlister.

Regler:

- Bevar demografi, personlighed og godkendte attributter uden at opfinde fakta.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder ud
  over den leverede kommune.
- Udled ikke religion, etnicitet, helbred, seksualitet, politik eller kriminalitet.
- Påstå ikke familie, husstand, diagnose eller fysisk udseende.
- Brug ikke `origin_country_da` til at udlede kultur, etnicitet, udseende, job,
  færdigheder eller interesser.
- Giv aldrig tidligere eller tidligere formuleret arbejde, heller ikke for ansatte:
  undgå `tidligere`, `førhen`, `arbejdede`, `har arbejdet`, `pensioneret fra` og
  `forhenværende`.
- Behandl OCEAN som svage tilbøjeligheder, aldrig som diagnoser, evner eller mangler.
- Omtal færdigheder som mulige interesser eller styrker, ikke dokumenteret erfaring.
- Brug naturlige danske ord og hold hvert specialiseret felt i sit eget domæne.
- De seks tekster skal være forskellige; gentag ikke en hel tekst ordret.
