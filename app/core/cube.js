const path = require('path');

const fs = require('fs');

const jwt = require('jsonwebtoken');





const normalizeTenantId = (value) =>

  String(value || 'default')

  .normalize('NFKD')

  .replace(/[\u0300-\u036f]/g, '')

  .toLowerCase()

  .replace(/[^a-z0-9_-]/g, '_')

  .slice(0, 100);



let devContext = null;

const resolveContext = (securityContext) => {

if (securityContext && securityContext.companyDbHost) return securityContext;

if (process.env.CUBEJS_DEV_MODE === 'true' && devContext) return devContext;

return securityContext || {};

};

// L'interface Data Model de Cube lit/écrit toujours dans "model" :
// on le fait pointer vers le dossier du tenant actif.
const linkModelDir = (tenantId) => 
  {  
    try 
    {    
      const targetRel = path.join(tenantId, 'model');    
      fs.mkdirSync(path.join(__dirname, targetRel, 'cubes'), { recursive: true });    
      const link = path.join(__dirname, 'model');    
      try {     
        if (!fs.lstatSync(link).isSymbolicLink())
           return; // vrai dossier : on n'y touche pas      
          fs.unlinkSync(link);    
        } 
      catch (e) {      
        if (e.code !== 'ENOENT') throw e;    
      }    
      fs.symlinkSync(targetRel, link, 'dir');  
    } catch (e) {    
      console.error('linkModelDir:', e.message);  
  }};

linkModelDir('default'); // évite l'erreur "Model files not found" au démarrage

// Mini serveur (dev uniquement) qui reçoit la connexion envoyée par FastAPI
if (process.env.CUBEJS_DEV_MODE === 'true') {
  const http = require('http');
  const port = Number(process.env.CUBE_DEV_CONTEXT_PORT || 4001);

  const server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    if (req.method !== 'POST' || req.url !== '/dev/tenant-context') {
      res.statusCode = 404;
      return res.end(JSON.stringify({ ok: false }));
    }
    const token = (req.headers.authorization || '').replace(/^Bearer\s+/i, '');
    try {
      const payload = jwt.verify(token, process.env.CUBEJS_API_SECRET, {
        algorithms: ['HS256'],
      });
      const { exp, iat, ...ctx } = payload;
      devContext = ctx;
      console.log('Dev context reçu pour le tenant:', ctx.tenantId, '| type:', ctx.companyDbType);
      linkModelDir(normalizeTenantId(ctx.tenantId));
      res.end(JSON.stringify({ ok: true, tenantId: ctx.tenantId }));
    } catch (e) {
      res.statusCode = 401;
      res.end(JSON.stringify({ ok: false, error: 'Token invalide.' }));
    }
  });

  server.on('error', (e) => console.error('Dev context server:', e.message));
  server.listen(port, '0.0.0.0', () =>
    console.log(`Dev context server listening on ${port}`)
  );
}

module.exports = {



// Désactive le Refresh Scheduler (il appelle driverFactory sans contexte)

scheduledRefreshTimer: false,
scheduledRefreshContexts: async () => [],





contextToAppId: ({ securityContext: rawContext }) => {

const securityContext = resolveContext(rawContext);

return `CUBE_APP_${normalizeTenantId(securityContext?.tenantId)}_${securityContext?.companyDbType || 'postgres'}`;

},

  contextToOrchestratorId: ({ securityContext: rawContext }) => {
    const securityContext = resolveContext(rawContext);

    const tenantId = normalizeTenantId(securityContext?.tenantId);

    const dbName = securityContext?.companyDbName || 'unconfigured';

    const dbType = securityContext?.companyDbType || 'postgres';

    return `TENANT_${tenantId}_${dbType}_${dbName}`;

    },



  driverFactory: ({ securityContext: rawContext }) => {
    const securityContext = resolveContext(rawContext);

    const required = [

    'companyDbType',

    'companyDbHost',

    'companyDbPort',

    'companyDbName',

    'companyDbUser',

    'companyDbPassword',

    ];

const missing = required.filter((key) => !securityContext?.[key]);

if (missing.length) {

throw new Error(`Missing tenant database settings: ${missing.join(', ')}`);

}



const port = Number.parseInt(securityContext.companyDbPort, 10);

if (!Number.isInteger(port) || port < 1 || port > 65535) {

throw new Error('Invalid tenant database port.');

}



return {

type: securityContext.companyDbType,

host: securityContext.companyDbHost,

port,

database: securityContext.companyDbName,

user: securityContext.companyDbUser,

password: securityContext.companyDbPassword,

poolMin: 1,

poolMax: 5,

};

},


  repositoryFactory: ({ securityContext: rawContext }) => {
    const securityContext = resolveContext(rawContext);

    const dbType = securityContext?.companyDbType || 'postgres';

    const tenantId = normalizeTenantId(securityContext?.tenantId);



    // Chemin dynamique basé sur le tenant_id (ex: /app/isnov/model/cubes)

    const schemaPath = path.join(__dirname, tenantId, 'model', 'cubes');



    // Sécurité : Crée le dossier dynamiquement s'il n'existe pas encore

    if (!fs.existsSync(schemaPath)) {

    fs.mkdirSync(schemaPath, { recursive: true });

    }



    return {

    localPath: () => schemaPath,

    dataSchemaFiles: () => {

    if (!fs.existsSync(schemaPath)) {

    return [];

    }



    const files = fs

    .readdirSync(schemaPath)

    .filter((file) => file.endsWith('.yaml') || file.endsWith('.yml'));



    return files.map((file) => ({

    // On intègre le tenantId pour que Cube.js isole bien les fichiers par entreprise

    fileName: path.join(tenantId, dbType, file),

    content: fs.readFileSync(path.join(schemaPath, file), 'utf-8'),

    }));

    },

    };

    },

    };