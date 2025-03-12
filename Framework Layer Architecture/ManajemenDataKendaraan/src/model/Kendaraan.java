package model;

public abstract class Kendaraan {
	private String id;
	private String merek;
	private int tahun;
	
//	ALT + SHIFT + S + O
	public Kendaraan(String id, String merek, int tahun) {
		super();
		this.id = id;
		this.merek = merek;
		this.tahun = tahun;
	}
	
//	ALT + SHIFT + S + R
	public String getId() {
		return id;
	}
	public String getMerek() {
		return merek;
	}
	public int getTahun() {
		return tahun;
	}
	
	public abstract String getTipe();
	
}
