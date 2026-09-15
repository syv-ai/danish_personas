# Danske baggrundsattributter

Du beskriver én opdigtet dansk person ud fra en fast profil med demografi og
personlighedsprofil. Personen er ikke virkelig, så beskriv hende eller ham som et
bestemt menneske, ikke som en mulighed.

Skriv naturligt dansk i nutid og returnér kun data, som passer til det krævede
JSON-skema. Demografi og personlighedsscorer er faste input og må ikke ændres.

Regler:

- Skriv konstaterende. Brug ikke forbehold som "muligvis", "måske", "kan tænkes",
  "har lettere ved", "formentlig", "sandsynligvis" eller "en tendens til".
- Attributterne er personens faktiske interesser og færdigheder, ikke sandsynlige bud.
- Vær konkret og specifik frem for generisk. Skriv "reparerer racercykler i kælderen"
  frem for "interesseret i sport".
- Attributterne skal være indbyrdes konsistente og logisk forbundne med profilen:
  alder, uddannelse, arbejdsmarkedsstatus og personlighed skal kunne ses i dem.
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
- Opfind ikke navn, adresse, arbejdsplads, uddannelsesinstitution eller
  kontaktoplysninger.
- Nævn ikke CPR-numre, telefonnumre, e-mailadresser, links eller præcise steder.
- Udled ikke religion, etnicitet, helbred, seksualitet, politisk overbevisning eller
  kriminalhistorik.
- Nævn ikke arbejdsgivere, titler eller certificeringer knyttet til virkelige
  organisationer.
- Brug persona-id'et som en neutral nøgle til variation, men gengiv det aldrig i svaret.
- Variér emnerne på tværs af personer; brug ikke automatisk standardkombinationen
  læsning, gåture og madlavning.
- Skriv 3-6 korte, forskellige færdigheder og 3-6 korte, forskellige interesser.
- Skriv et konkret karrieremål. Brug kun null, hvis personen står uden for
  arbejdsstyrken, og et mål derfor ikke giver mening.
- Den kulturelle kontekst skal være generel og ikke identificerende. Gengiv højst den
  angivne region; opfind ikke bystørrelse, lokale tilbud eller andre stedsegenskaber.
