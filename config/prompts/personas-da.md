# Danske personabeskrivelser

Du skriver seks korte, syntetiske tekster på naturligt dansk ud fra faste input:
`professional_persona`, `sports_persona`, `arts_persona`, `travel_persona`,
`culinary_persona` og `persona`. Returnér kun JSON efter skemaet. Hver tekst er
sammenhængende prosa på 2-4 sætninger. `persona` er højst 600 tegn og er beregnet til
senere billedprompting; den må ikke være en visuel beskrivelse.

`persona` skal indeholde hver værdi i `required_persona_facts` ordret i naturlig
prosa. Det gælder alle seks værdier: alder i formen `<n> år`, det danske statistiske
kønsord, kommuneetiketten, den brede danske uddannelsesfrase, oprindelsesetiketten og
det aktuelle arbejdsforhold. Værdierne skal gengives bogstaveligt: oversæt,
normalisér, bøj, uddyb eller pynt aldrig på dem, og skriv dem ikke som en liste. Brug
den leverede jobtitel for en ansat, ellers den leverede kanoniske statusfrase.
`education_level` er en sammenlagt kategori: udled eller opfind aldrig et mere
specifikt uddannelsesniveau eller en uddannelsesinstitution. Returnér altid alle
skemafelter; ingen felter må udelades. Indlejr nøjagtigt 2-3 forskellige interesser
ordret fra `generated_attributes`. Medtag nøjagtigt 1-2 forskellige OCEAN-tendenser
som bogstavelige termer fra den leverede liste `allowed_personality_tendencies`, og
brug ikke andre personlighedstermer. Hver term skal hedges i nærheden i samme
ledsætning med et af de tilladte ord `kan`, `ofte`, `muligvis`, `gerne` eller `typisk`.
Rå OCEAN-scorer og labels hører ikke til dette trin: fortolk dem ikke, og opfind ikke
andre personlighedstermer end den leverede liste. Skriv aldrig punktlister,
nummerering, parenteser, klammer eller semikolonlister.

Regler:

- Bevar demografi, personlighed og godkendte attributter uden at opfinde fakta.
- Opfind ikke navn, adresse, kontaktoplysninger, arbejdsplads eller institution.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder ud
  over den leverede kommune.
- Udled ikke religion, etnicitet, helbred, seksualitet, politik eller kriminalitet.
- Påstå ikke familie, husstand, diagnose eller fysisk udseende.
- Giv aldrig tidligere eller tidligere formuleret arbejde, heller ikke for ansatte:
  undgå `tidligere`, `førhen`, `arbejdede`, `har arbejdet`, `pensioneret fra` og
  `forhenværende`.
- Behandl OCEAN som svage tilbøjeligheder, aldrig som diagnoser, evner eller mangler.
- Omtal færdigheder som mulige interesser eller styrker, ikke dokumenteret erfaring.
- Brug naturlige danske ord og hold hvert specialiseret felt i sit eget domæne.
- De seks tekster skal være forskellige; gentag ikke en hel tekst ordret.
