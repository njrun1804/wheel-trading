-- Fix column name mismatches between code and database

-- Check if columns exist before renaming
DO $$ 
BEGIN
    -- Options table
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_schema='options' AND table_name='contracts' AND column_name='strike') 
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_schema='options' AND table_name='contracts' AND column_name='strike_price') THEN
        ALTER TABLE options.contracts RENAME COLUMN strike TO strike_price;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_schema='options' AND table_name='contracts' AND column_name='bid') 
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_schema='options' AND table_name='contracts' AND column_name='bid_price') THEN
        ALTER TABLE options.contracts RENAME COLUMN bid TO bid_price;
    END IF;
    
    IF EXISTS (SELECT 1 FROM information_schema.columns 
               WHERE table_schema='options' AND table_name='contracts' AND column_name='ask') 
       AND NOT EXISTS (SELECT 1 FROM information_schema.columns 
                       WHERE table_schema='options' AND table_name='contracts' AND column_name='ask_price') THEN
        ALTER TABLE options.contracts RENAME COLUMN ask TO ask_price;
    END IF;
END $$;

-- Create views for backward compatibility
CREATE OR REPLACE VIEW options.contracts_compat AS
SELECT 
    *,
    strike_price as strike,
    bid_price as bid,
    ask_price as ask
FROM options.contracts;

-- Grant permissions
GRANT SELECT ON options.contracts_compat TO PUBLIC;
