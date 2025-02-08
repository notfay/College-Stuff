package model;

public abstract class Pekerjaan {
	//Parent Class
	
	private String NamaPekerjaan;
	private int JamMasuk;
	private int JamKeluar;
	private String Deskripsi;
	
	//Constructor Alt+Shift+S+O
	
	public Pekerjaan(String namaPekerjaan, int jamMasuk, int jamKeluar, String deskripsi) {
		super();
		NamaPekerjaan = namaPekerjaan;
		JamMasuk = jamMasuk;
		JamKeluar = jamKeluar;
		Deskripsi = deskripsi;
	}

	
	//Getter Setter Method, Alt+Shift+S+R
	
	
	public String getNamaPekerjaan() {
		return NamaPekerjaan;
	}

	public void setNamaPekerjaan(String namaPekerjaan) {
		NamaPekerjaan = namaPekerjaan;
	}

	public int getJamMasuk() {
		return JamMasuk;
	}

	public void setJamMasuk(int jamMasuk) {
		JamMasuk = jamMasuk;
	}

	public int getJamKeluar() {
		return JamKeluar;
	}

	public void setJamKeluar(int jamKeluar) {
		JamKeluar = jamKeluar;
	}

	public String getDeskripsi() {
		return Deskripsi;
	}

	public void setDeskripsi(String deskripsi) {
		Deskripsi = deskripsi;
	}
	
	
	//-------------------------------------------------------//
	
	public abstract void BerangkatKerja(); 	//Ctrl+Spasi -> Auto
		
	
	
}
