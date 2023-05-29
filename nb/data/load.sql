-- create database movfefe;
use movfefe;

create table movies (
    Title varchar(100),
    US_Gross float,
    Worldwide_Gross float,
    US_DVD_Sales int,
    Production_Budget int,
    -- data is in 'Jan 1 2000' format, use varchar to load.
    -- chatgpt tells us that we can convert it to date using
    -- str_to_date(Release_Date, '%b %d %Y')
    Release_Date varchar(50), 
    MPAA_Rating varchar(5),
    Running_Time_min int,
    Distributor varchar(50),
    Source varchar(50),
    Major_Genre varchar(50),
    Creative_Type varchar(50),
    Director varchar(50),
    Rotten_Tomatoes_Rating float,
    IMDB_Rating float,
    IMDB_Votes int
);

-- use absolute path
-- load data infile './movies.csv' into table movies;

select count(*) from movies;
