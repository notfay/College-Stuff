package proxy;

import java.util.HashMap;

import model.Kendaraan;

public class InfoProxy implements InfoKendaraan {
	
	private InfoDB info = new InfoDB();
	private HashMap<String, String> cache = new HashMap<>();
	
	@Override
	public String getInfo(Kendaraan kendaraan) {
		String key = kendaraan.getNama() + kendaraan.getHarga() + kendaraan.getTipe();
		
		if(!cache.containsKey(key)) {
			System.out.println("Ambil data dari Database");

			String informasi = info.getInfo(kendaraan);

			cache.put(key, informasi);
		} 
		
		else {
			System.out.println("Ambil data dari Cache");
		}
		
		return cache.get(key);
	}
	

}