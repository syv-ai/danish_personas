# Reparation af personaer

Skriv en sammenhængende dansk persona på 300–900 tegn ud fra de medsendte,
syntetiske fakta. Returnér kun et JSON-objekt med feltet `persona`. Behold alder,
arbejde, uddannelse og sted konsistent med input; opfind ikke nye personlige
oplysninger eller ændrede strukturerede attributter.

`gender` og `partner_gender` er syntetiske, og kan være `man`, `woman`,
`nonbinary` eller `unknown`; partnerens køn kan også være null. Omtal personen
og en eventuel partner i overensstemmelse med disse felter, men undgå at
antage køn, når værdien er ukendt. En ikke-partneret person har ingen aktuel
partner. Hvis `legal_status_detail` mangler, må du ikke gætte, om personen
er gift eller separeret. Undlad hellere en usikker oplysning end at opfinde en.

Udled ikke seksualitet, transstatus, variationer i kønskarakteristika,
statsborgerskab, religion, udseende eller kulturelle interesser af køn,
partner, oprindelsesland eller bopæl. Omtal ikke interne måltal eller
kilde- og validitetsoplysninger.
